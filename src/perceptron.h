#pragma once

#include <vector>

#include "math/matrix.h"

namespace NeuralNetwork
{
    template<typename T>
    class Perceptron
    {
    private:
        std::vector<Math::Matrix<T>> _layers;
        std::vector<Math::Matrix<T>> _weights;
        std::vector<Math::Matrix<T>> _bias;
        
        std::vector<Math::Matrix<T>> _derivatives;
        std::vector<Math::Matrix<T>> _deltas;
        std::vector<Math::Matrix<T>> _deltasWeights;
        std::vector<Math::Matrix<T>> _deltasBias;

        std::vector<Math::Matrix<T>> _deltasWeightsInertia;
        std::vector<Math::Matrix<T>> _deltasBiasInertia;

        bool _cacheIsInitialized;

    public:
        Perceptron(const std::vector<int>& neuronsCountPerLayer);

        void RandomizeWeights(unsigned int seed, T lowerBorder, T upperBorder);

        void SetInputValues(const Math::Matrix<T>& inputValues);

        const Math::Matrix<T>& ForwardPropagation(T(*activationFunction)(T));

        const Math::Matrix<T>& ForwardPropagationWithCache(T(*activationFunction)(T), T(*derivativeFunction)(T), bool cacheAfterActivationFunction = false);
        void BackwardPropagation(const Math::Matrix<T>& idealValues, T learningRate, T moment);

        void InitTrainCache();
        void ClearTrainCache();
    };
}