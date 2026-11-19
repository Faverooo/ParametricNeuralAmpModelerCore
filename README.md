# ParametricNAM Core

The audio engine behind the [ParametricNAM VST3](https://github.com/Faverooo/ParametricNeuralAmpModelerPlugin). It loads `.pnam` amp models and uses their knob settings to process guitar audio. This project is based on Steven Atkinson’s [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore).

## Use it

**To play a model**, install the [VST3](https://github.com/Faverooo/ParametricNeuralAmpModelerPlugin/releases/latest). It already includes the Core.

**To use the engine in your own C++ project**, clone this repository with its submodules. Start with `nam::ParametricModel::Load` in [NAM/pnam.h](NAM/pnam.h). The example test loads a model, changes a knob and processes audio. [C++ example](tools/test_pnam.cpp)

## License

The Core is [MIT licensed](LICENSE). Third-party dependencies keep their own license terms. Thanks to [TONE3000](https://tone3000.com) for sponsoring development of the upstream Core.
