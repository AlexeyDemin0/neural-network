#include "perceptron.h"

#include <stdexcept>

namespace NeuralNetwork
{
    template<typename T>
    Perceptron<T>::Perceptron(const std::vector<int>& neuronsCountPerLayer) : _cacheIsInitialized(false)
    {
        if (neuronsCountPerLayer.size() < 1)
            throw std::invalid_argument("Neuron layers count must be more than 1");

        _layers.resize(neuronsCountPerLayer.size());
        for (int i = 0; i < neuronsCountPerLayer.size(); i++)
        {
            _layers[i] = Math::Matrix<T>(neuronsCountPerLayer[i], 1, false);
        }

        _weights.resize(neuronsCountPerLayer.size() - 1);
        _bias.resize(neuronsCountPerLayer.size() - 1);
        for (int i = 0; i < neuronsCountPerLayer.size() - 1; i++)
        {
            int neuronsCountCurrent = neuronsCountPerLayer[i];
            int neuronsCountNext = neuronsCountPerLayer[i + 1];
            _weights[i] = Math::Matrix<T>(neuronsCountNext, neuronsCountCurrent, false);
            _bias[i] = Math::Matrix<T>(neuronsCountNext, 1, false);
        }
    }

    template<typename T>
    void Perceptron<T>::RandomizeWeights(unsigned int seed, T lowerBorder, T upperBorder)
    {
        srand(seed);
        T dist = upperBorder - lowerBorder;

        for (int i = 0; i < _weights.size(); i++)
        {
            for (int row = 0; row < _weights[i].GetRows(); row++)
            {
                for (int col = 0; col < _weights[i].GetCols(); col++)
                {
                    T randValue = rand() / static_cast<T>(RAND_MAX);
                    _weights[i](row, col) = randValue * dist + lowerBorder;
                }
            }

            for (int row = 0; row < _bias[i].GetRows(); row++)
            {
                T randValue = rand() / static_cast<T>(RAND_MAX);
                _bias[i](row, 0) = randValue * dist + lowerBorder;
            }
        }
    }

    template<typename T>
    void Perceptron<T>::SetInputValues(const Math::Matrix<T>& inputValues)
    {
        _layers[0] = inputValues;
    }

    //
    // Algorithm of forward propagation:
    // [LaTeX-like syntax]:
    //      z^l = W^l * a^{l-1} + b^l
    //      a^l = \sigma(z^l)
    // where:
    //      l - layer index.
    //      z^l - weighted sum of the neuron inputs.
    //      \sigma(x) - activation function.  
    //      W^l - matrix of weights between layers (l) and (l-1), dimension are R^{N(l)xN(l-1)} 
    //          where N(l) is the number of neurons in layer (l).
    //      b^l - bias term for the layer.
    // Note:
    //      Indexes in code may not match.
    //
    template<typename T>
    const Math::Matrix<T>& Perceptron<T>::ForwardPropagation(T(*activationFunction)(T))
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

    //
    // This is forward propagation with saving derivatives for use in backward propagation.
    // Param @cacheAfterActivationFunction is used to save the derivative after the activation function, 
    //      which is helpful for calculating the derivative of the Sigmoid or Hyperbolic Tangent functions.
    // 
    template<typename T>
    const Math::Matrix<T>& Perceptron<T>::ForwardPropagationWithCache(T(*activationFunction)(T), T(*derivativeFunction)(T), bool cacheAfterActivationFunction)
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
                _derivatives[i] = _layers[i + 1];
                _derivatives[i].ApplyFunction(derivativeFunction);
            }
            else
            {
                _derivatives[i] = _layers[i + 1];
                _derivatives[i].ApplyFunction(derivativeFunction);
                _layers[i + 1].ApplyFunction(activationFunction);
            }
        }
        return _layers[_layers.size() - 1];
    }

    //
    // Algorithm of backward propagation:
    // [LaTeX-like syntax]:
    //      Delta calculation for last layer (L):
    //      \delta^l = dL/da^L \odot \sigma'(z^L)
    //      
    //      Delta calculation for hidden layers (l):
    //      \delta^l = (W^{l+1})^T * \delta^{l+1} \odot \sigma'(z^l)
    // 
    //      Calculation of partial derivatives of L:
    //      dL/dW^l = \delta^l * (a^{l-1})^T
    //      dL/db^l = \delta^l
    // 
    //      Weights adjustment:
    //      W^l <- W^l - k * dL/dW^l
    //      b^l <- b^l - k * dL/db^l
    // where:
    //      [the same definitions as in forward propagation]
    //      \delta^l - gradient of the loss function with respect to the weighted sums at layer l.
    //      dL/dW^l - gradient of the loss function with respect to the weights at layers between (l) and (l-1).
    //      dL/db^l - gradient of the loss function with respect to the bias term at layers between (l) and (l-1).
    //      L - loss function (MSE in current implementation)
    //      k - learning rate coefficient
    // Note:
    //      Indexes in code may not match.
    //
    template<typename T>
    void Perceptron<T>::BackwardPropagation(const Math::Matrix<T>& idealValues, T learningRate, T moment)
    {
        int layerIndex = _layers.size() - 2;
        _deltas[layerIndex] = _layers[layerIndex + 1];
        _deltas[layerIndex] -= idealValues;
        _deltas[layerIndex] *= static_cast<T>(2.0);
        _deltas[layerIndex].HadamardProductThis(_derivatives[layerIndex]);

        Math::Matrix<T>::MultMatrixToTransposedAndStoreTo(_deltas[layerIndex], _layers[layerIndex], _deltasWeights[layerIndex]);
        _deltasBias[layerIndex] = _deltas[layerIndex];

        _deltasWeightsInertia[layerIndex] *= moment;
        _deltasBiasInertia[layerIndex] *= moment;
        _deltasWeights[layerIndex] *= (static_cast<T>(1.0) - moment);
        _deltasBias[layerIndex] *= (static_cast<T>(1.0) - moment);
        _deltasWeightsInertia[layerIndex] += _deltasWeights[layerIndex];
        _deltasBiasInertia[layerIndex] += _deltasBias[layerIndex];

        layerIndex--;

        // Hidden layers
        for (; layerIndex >= 0; layerIndex--)
        {
            Math::Matrix<T>::MultTransposedToMatrixAndStoreTo(_weights[layerIndex + 1], _deltas[layerIndex + 1], _deltas[layerIndex]);
            _deltas[layerIndex].HadamardProductThis(_derivatives[layerIndex]);

            Math::Matrix<T>::MultMatrixToTransposedAndStoreTo(_deltas[layerIndex], _layers[layerIndex], _deltasWeights[layerIndex]);
            _deltasBias[layerIndex] = _deltas[layerIndex];

            _deltasWeightsInertia[layerIndex] *= moment;
            _deltasBiasInertia[layerIndex] *= moment;
            _deltasWeights[layerIndex] *= (static_cast<T>(1.0) - moment);
            _deltasBias[layerIndex] *= (static_cast<T>(1.0) - moment);
            _deltasWeightsInertia[layerIndex] += _deltasWeights[layerIndex];
            _deltasBiasInertia[layerIndex] += _deltasBias[layerIndex];
        }

        // Weights adjusting
        for (int weightIndex = 0; weightIndex < _weights.size(); weightIndex++)
        {
            _deltasWeights[weightIndex].MultAndStoreThis(_deltasWeightsInertia[weightIndex], learningRate);
            _deltasBias[weightIndex].MultAndStoreThis(_deltasBiasInertia[weightIndex], learningRate);

            _weights[weightIndex] -= _deltasWeights[weightIndex];
            _bias[weightIndex] -= _deltasBias[weightIndex];
        }
    }

    template<typename T>
    void Perceptron<T>::InitTrainCache()
    {
        if (_cacheIsInitialized)
            ClearTrainCache();

        _cacheIsInitialized = true;

        int layersCount = _layers.size();
        _derivatives.resize(layersCount - 1);
        _deltas.resize(layersCount - 1);
        _deltasWeights.resize(layersCount - 1);
        _deltasBias.resize(layersCount - 1);
        _deltasWeightsInertia.resize(layersCount - 1);
        _deltasBiasInertia.resize(layersCount - 1);

        for (int i = 0; i < layersCount - 1; i++)
        {
            int neuronsCountCurrent = _layers[i].GetRows();
            int neuronsCountNext = _layers[i + 1].GetRows();

            _derivatives[i] = Math::Matrix<T>(_layers[i + 1].GetRows(), 1, false);
            _deltas[i] = Math::Matrix<T>(_layers[i + 1].GetRows(), 1, false);
            _deltasWeights[i] = Math::Matrix<T>(neuronsCountNext, neuronsCountCurrent, false);
            _deltasBias[i] = Math::Matrix<T>(neuronsCountNext, 1, false);
            _deltasWeightsInertia[i] = Math::Matrix<T>(neuronsCountNext, neuronsCountCurrent);
            _deltasBiasInertia[i] = Math::Matrix<T>(neuronsCountNext, 1);
        }
    }

    template<typename T>
    void Perceptron<T>::ClearTrainCache()
    {
        _cacheIsInitialized = false;
        _derivatives.clear();
        _deltas.clear();
        _deltasWeights.clear();
        _deltasBias.clear();
        _deltasWeightsInertia.clear();
        _deltasBiasInertia.clear();
    }

    template class Perceptron<float>;
    template class Perceptron<double>;
}