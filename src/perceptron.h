#include <vector>

#include "math/matrix.h"

#define NN_TYPE float

namespace NeuralNetwork
{
    class Perceptron
    {
    private:
        std::vector<Math::Matrix<NN_TYPE>> _layers;
        std::vector<Math::Matrix<NN_TYPE>> _weights;
        std::vector<Math::Matrix<NN_TYPE>> _bias;
        
        std::vector<Math::Matrix<NN_TYPE>> _derivatives;
        std::vector<Math::Matrix<NN_TYPE>> _deltas;
        std::vector<Math::Matrix<NN_TYPE>> _deltasWeights;
        std::vector<Math::Matrix<NN_TYPE>> _deltasBias;

        bool _cacheIsInitialized;

    public:
        Perceptron(std::vector<int> neuronsCountPerLayer);

        void SetInputValues(const Math::Matrix<NN_TYPE>& inputValues);

        const Math::Matrix<NN_TYPE>& ForwardPropagation(NN_TYPE(*activationFunction)(NN_TYPE));

        const Math::Matrix<NN_TYPE>& ForwardPropagationWithCache(NN_TYPE(*activationFunction)(NN_TYPE), NN_TYPE(*derivativeFunction)(NN_TYPE), bool cacheAfterActivationFunction = false);
        void BackwardPropagation(const Math::Matrix<NN_TYPE>& idealValues, NN_TYPE learningRate);

        void InitTrainCache();
        void ClearTrainCache();
    };
}