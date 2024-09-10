#include "perceptron.h"

#include <stdexcept>

namespace NeuralNetwork
{
    Perceptron::Perceptron(const std::vector<int>& neuronsCountPerLayer) : _cacheIsInitialized(false)
    {
        if (neuronsCountPerLayer.size() < 1)
            throw std::invalid_argument("Neuron layers count must be more than 1");

        _layers.resize(neuronsCountPerLayer.size());
        for (int i = 0; i < neuronsCountPerLayer.size(); i++)
        {
            _layers[i] = Math::Matrix<NN_TYPE>(neuronsCountPerLayer[i], 1, false);
        }

        _weights.resize(neuronsCountPerLayer.size() - 1);
        _bias.resize(neuronsCountPerLayer.size() - 1);
        for (int i = 0; i < neuronsCountPerLayer.size() - 1; i++)
        {
            int neuronsCount = neuronsCountPerLayer[i];
            int neuronsCountNext = neuronsCountPerLayer[i + 1];
            _weights[i] = Math::Matrix<NN_TYPE>(neuronsCountNext, neuronsCount, false);
            _bias[i] = Math::Matrix<NN_TYPE>(neuronsCountNext, 1, false);
        }
    }

    void Perceptron::SetInputValues(const Math::Matrix<NN_TYPE>& inputValues)
    {
        _layers[0] = inputValues;
    }

    const Math::Matrix<NN_TYPE>& Perceptron::ForwardPropagation(NN_TYPE(*activationFunction)(NN_TYPE))
    {
        for (int i = 0; i < _layers.size() - 1; i++)
        {
            _layers[i + 1]
                .MultAndStoreThis(_weights[i], _layers[i])
                .AddCol(_bias[i], 0)
                .ApplyFunction(activationFunction);
        }
        return _layers[_layers.size() - 1];
    }

    const Math::Matrix<NN_TYPE>& Perceptron::ForwardPropagationWithCache(NN_TYPE(*activationFunction)(NN_TYPE), NN_TYPE(*derivativeFunction)(NN_TYPE), bool cacheAfterActivationFunction)
    {
        if (!_cacheIsInitialized)
            throw std::exception("Cache is not initialized. Use InitTrainCache() method.");

        for (int i = 0; i < _layers.size() - 1; i++)
        {
            _layers[i + 1]
                .MultAndStoreThis(_weights[i], _layers[i])
                .AddCol(_bias[i], 0);

            if (cacheAfterActivationFunction)
            {
                _layers[i + 1].ApplyFunction(activationFunction);
                _derivatives[i + 1] = _layers[i + 1];
                _derivatives[i + 1].ApplyFunction(derivativeFunction);
            }
            else
            {
                _derivatives[i + 1] = _layers[i + 1];
                _derivatives[i + 1].ApplyFunction(derivativeFunction);
                _layers[i + 1].ApplyFunction(activationFunction);
            }
        }
        return _layers[_layers.size() - 1];
    }

    void Perceptron::BackwardPropagation(const Math::Matrix<NN_TYPE>& idealValues, NN_TYPE learningRate)
    {
        // Output layer
        int layerIndex = _layers.size() - 1;
        _deltas[layerIndex] = _layers[layerIndex];
        _deltas[layerIndex] -= idealValues;
        _deltas[layerIndex] *= static_cast<NN_TYPE>(2.0);
        _deltas[layerIndex].HadamardProductThis(_derivatives[layerIndex]);

        Math::Matrix<NN_TYPE>::MultMatrixToTransposedAndStoreTo(_deltas[layerIndex], _layers[layerIndex - 1], _deltasWeights[layerIndex - 1]);
        _deltasBias[layerIndex - 1] = _deltas[layerIndex];
        layerIndex--;

        // Hidden layers
        for (; layerIndex > 0; layerIndex--)
        {
            _deltas[layerIndex + 1].HadamardProductThis(_derivatives[layerIndex]);
            Math::Matrix<NN_TYPE>::MultTransposedToMatrixAndStoreTo(_weights[layerIndex - 1], _deltas[layerIndex + 1], _deltas[layerIndex]);

            Math::Matrix<NN_TYPE>::MultMatrixToTransposedAndStoreTo(_deltas[layerIndex], _layers[layerIndex - 1], _deltasWeights[layerIndex - 1]);
            _deltasBias[layerIndex - 1] = _deltas[layerIndex];
        }

        // Weights adjusting
        for (int weightIndex = 0; weightIndex < _weights.size(); weightIndex++)
        {
            _deltasWeights[weightIndex] *= learningRate;
            _deltasBias[weightIndex] *= learningRate;

            _weights[weightIndex] -= _deltasWeights[weightIndex];
            _bias[weightIndex] -= _deltasBias[weightIndex];
        }
    }

    void Perceptron::InitTrainCache()
    {
        if (_cacheIsInitialized)
            ClearTrainCache();

        _cacheIsInitialized = true;

        int layersCount = _layers.size();
        _derivatives.reserve(layersCount);
        _deltas.reserve(layersCount);
        _deltasWeights.reserve(layersCount - 1);
        _deltasBias.reserve(layersCount - 1);

        for (int i = 0; i < layersCount; i++)
        {
            _derivatives[i] = Math::Matrix<NN_TYPE>(_layers[i].GetRows(), 1, false);
            _deltas[i] = Math::Matrix<NN_TYPE>(_layers[i].GetRows(), 1, false);
        }

        for (int i = 0; i < layersCount - 1; i++)
        {
            int neuronsCount = _layers[i].GetRows();
            int neuronsCountNext = _layers[i + 1].GetRows();
            _deltasWeights[i] = Math::Matrix<NN_TYPE>(neuronsCountNext, neuronsCount, false);
            _deltasBias[i] = Math::Matrix<NN_TYPE>(neuronsCountNext, 1, false);
        }
    }

    void Perceptron::ClearTrainCache()
    {
        _cacheIsInitialized = false;
        _derivatives.clear();
        _deltas.clear();
        _deltasWeights.clear();
        _deltasBias.clear();
    }
}