#include "NAM/get_dsp.h"
#include "NAM/pnam.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
void check(bool result, const char* message)
{
  if (!result)
    throw std::runtime_error(message);
}

std::array<NAM_SAMPLE, 128> render(nam::DSP& dsp)
{
  std::array<NAM_SAMPLE, 128> input{}, output{};
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = NAM_SAMPLE(0.1 * std::sin(0.17 * double(i)));

  dsp.Reset(48000, 64);
  for (size_t offset = 0; offset < input.size(); offset += 64)
  {
    auto* in = input.data() + offset;
    auto* out = output.data() + offset;
    dsp.process(&in, &out, 64);
  }
  return output;
}
} // namespace

int main(int argc, char** argv)
{
  try
  {
    check(argc == 3, "Usage: test_pnam example.nam temporary.pnam");
    std::ifstream input(argv[1]);
    const auto base = nlohmann::json::parse(input);
    nlohmann::json layers = nlohmann::json::array();
    for (const auto& group : base.at("config").at("layers"))
    {
      const size_t channels = group.at("channels");
      const std::vector<std::vector<float>> scale(channels, {0.25f});
      const std::vector<std::vector<float>> shift(channels, {0.2f});
      const std::vector<float> bias(channels, 0.0f);
      for (size_t i = 0; i < group.at("dilations").size(); ++i)
        layers.push_back({{"scale", scale}, {"shift", shift}, {"bias", bias}});
    }

    const nlohmann::json controls = nlohmann::json::array(
      {{{"id", "gain"}, {"label", "Gain"}, {"min", 2}, {"max", 10}, {"default", 2}, {"model_min", 0}, {"model_max", 1}}});
    const nlohmann::json document = {{"format", "pnam"}, {"version", 1}, {"amplifier", "CI amp"},
                                    {"controls", controls}, {"base_nam", base}, {"film_linear", layers}};
    {
      std::ofstream output(argv[2]);
      check(bool(output << document), "Cannot write PNAM test fixture");
    }

    auto model = nam::ParametricModel::Load(argv[2]);
    auto stock = nam::get_dsp(base);
    check(model.Amplifier() == "CI amp" && model.Controls().size() == 1, "Wrong PNAM metadata");
    const auto reference = render(*stock);
    const auto initial = render(model.Audio());
    for (size_t i = 0; i < initial.size(); ++i)
      check(std::abs(initial[i] - reference[i]) < 1e-6, "Neutral FiLM differs from stock WaveNet");

    check(!model.SetControls(std::array<float, 1>{11.0f}), "Invalid dial accepted");
    check(model.SetControls(std::array<float, 1>{10.0f}), "Valid dial rejected");
    const auto changed = render(model.Audio());
    double maxDifference = 0;
    for (size_t i = 0; i < changed.size(); ++i)
    {
      check(std::isfinite(changed[i]), "Nonfinite output");
      maxDifference = std::max(maxDifference, double(std::abs(changed[i] - initial[i])));
    }
    check(maxDifference > 1e-5, "FiLM did not affect the output");
    std::cout << "PNAM loader, dial mapping and WaveNet render passed\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
