#pragma once

#include <span>

#include "get_dsp.h"
#include "wavenet/model.h"

namespace nam
{
struct ParametricControl
{
  std::string id;
  std::string label;
  float minimum;
  float maximum;
  float initial;
  float modelMinimum;
  float modelMaximum;
  std::optional<float> captureStep;
};

class ParametricModel
{
public:
  static ParametricModel Load(const std::filesystem::path& path);

  DSP& Audio() { return *_audio; }
  const std::string& Amplifier() const { return _amplifier; }
  const std::vector<ParametricControl>& Controls() const { return _controls; }

  /// Set physical dial values between audio blocks on the processing thread.
  /// Returns false for invalid input without changing the DSP coefficients.
  bool SetControls(std::span<const float> values);

private:
  ParametricModel(std::unique_ptr<wavenet::WaveNet> audio, std::string amplifier,
                  std::vector<ParametricControl> controls);

  std::unique_ptr<wavenet::WaveNet> _audio;
  std::string _amplifier;
  std::vector<ParametricControl> _controls;
  Eigen::VectorXf _modelValues;
};
} // namespace nam
