#include "pnam.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_set>

namespace
{
using nlohmann::json;

Eigen::MatrixXf ReadMatrix(const json& rows, size_t controls)
{
  if (!rows.is_array() || rows.empty())
    throw std::invalid_argument("Expected a nonempty FiLM matrix");
  Eigen::MatrixXf matrix(rows.size(), controls);
  for (size_t row = 0; row < rows.size(); ++row)
  {
    if (!rows[row].is_array() || rows[row].size() != controls)
      throw std::invalid_argument("Invalid FiLM matrix width");
    for (size_t col = 0; col < controls; ++col)
      matrix(row, col) = rows[row][col].get<float>();
  }
  return matrix;
}

float ReadFinite(const json& object, const char* key)
{
  const auto value = object.at(key).get<float>();
  if (!std::isfinite(value))
    throw std::invalid_argument(std::string("Nonfinite PNAM value: ") + key);
  return value;
}
} // namespace

nam::ParametricModel::ParametricModel(std::unique_ptr<wavenet::WaveNet> audio, std::string amplifier,
                                      std::vector<ParametricControl> controls)
: _audio(std::move(audio))
, _amplifier(std::move(amplifier))
, _controls(std::move(controls))
, _modelValues(_controls.size())
{
}

nam::ParametricModel nam::ParametricModel::Load(const std::filesystem::path& path)
{
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Cannot open PNAM file: " + path.string());
  const auto document = json::parse(file);
  if (document.at("format") != "pnam" || document.at("version") != 1)
    throw std::invalid_argument("Unsupported PNAM format or version");

  std::vector<ParametricControl> controls;
  std::unordered_set<std::string> ids;
  const auto& descriptors = document.at("controls");
  if (!descriptors.is_array() || descriptors.empty())
    throw std::invalid_argument("PNAM requires at least one control");
  Eigen::VectorXf initialValues(descriptors.size());
  for (const auto& entry : descriptors)
  {
    ParametricControl control{
      entry.at("id").get<std::string>(), entry.at("label").get<std::string>(),
      ReadFinite(entry, "min"), ReadFinite(entry, "max"), ReadFinite(entry, "default"),
      ReadFinite(entry, "model_min"), ReadFinite(entry, "model_max"), std::nullopt};
    if (entry.contains("capture_step"))
      control.captureStep = ReadFinite(entry, "capture_step");
    if (control.id.empty() || control.label.empty() || !ids.insert(control.id).second
        || control.minimum >= control.maximum || control.initial < control.minimum || control.initial > control.maximum
        || control.modelMinimum >= control.modelMaximum || (control.captureStep && *control.captureStep <= 0.0f))
      throw std::invalid_argument("Invalid PNAM control descriptor");
    initialValues(controls.size()) =
      std::lerp(control.modelMinimum, control.modelMaximum,
                (control.initial - control.minimum) / (control.maximum - control.minimum));
    controls.push_back(std::move(control));
  }

  std::vector<wavenet::detail::KnobFilm> films;
  const auto& layers = document.at("film_linear");
  if (!layers.is_array())
    throw std::invalid_argument("Expected ordered PNAM FiLM layers");
  for (const auto& layer : layers)
  {
    const auto bias = layer.at("bias").get<std::vector<float>>();
    films.push_back({ReadMatrix(layer.at("scale"), controls.size()), ReadMatrix(layer.at("shift"), controls.size()),
                     Eigen::Map<const Eigen::VectorXf>(bias.data(), bias.size()), {}, {}});
  }

  const auto& base = document.at("base_nam");
  if (base.at("architecture") != "WaveNet" || ReadFinite(base, "sample_rate") <= 0.0f)
    throw std::invalid_argument("PNAM requires a WaveNet with a positive sample rate");
  auto audio = get_dsp(base, DspLoadOptions{false});
  auto* waveNet = dynamic_cast<wavenet::WaveNet*>(audio.get());
  if (!waveNet)
    throw std::invalid_argument("PNAM requires the generic WaveNet implementation");
  waveNet->ConfigureKnobFilm(std::move(films), initialValues);
  auto amplifier = document.at("amplifier").get<std::string>();
  audio.release();
  return ParametricModel(std::unique_ptr<wavenet::WaveNet>(waveNet), std::move(amplifier), std::move(controls));
}

bool nam::ParametricModel::SetControls(std::span<const float> values)
{
  if (values.size() != _controls.size())
    return false;
  for (size_t i = 0; i < values.size(); ++i)
  {
    const auto& control = _controls[i];
    if (!std::isfinite(values[i]) || values[i] < control.minimum || values[i] > control.maximum)
      return false;
    _modelValues(i) = std::lerp(control.modelMinimum, control.modelMaximum,
                                (values[i] - control.minimum) / (control.maximum - control.minimum));
  }
  _audio->SetKnobValues(_modelValues);
  return true;
}
