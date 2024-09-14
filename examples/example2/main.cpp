//
// Example 2: XOR logical gate
//

#include <iostream>
#include <cstdio>
#include <vector>
#include <ctime>

#include "perceptron.h"
#include "math/matrix.h"
#include "math/functions.h"

#define STEPS 1000
#define LEARNING_RATE_LINEAR 0.05f
#define LEARNING_RATE_NONLINEAR 0.1f
#define MOMENT 0.8f

using namespace NeuralNetwork;
using namespace NeuralNetwork::Math;
using namespace NeuralNetwork::Math::Functions;

int main()
{
    std::vector<int> neuronsPerLayer_Linear = { 2, 5, 5, 5, 1 };
    std::vector<int> neuronsPerLayer_NonLinear = { 2, 2, 1 };
    Perceptron<float> perceptron_Linear(neuronsPerLayer_Linear);
    Perceptron<float> perceptron_NonLinear(neuronsPerLayer_NonLinear);

    std::vector<Matrix<float>> inputs;
    std::vector<Matrix<float>> outputs;

    inputs.push_back({ {0}, {0} });
    inputs.push_back({ {1}, {0} });
    inputs.push_back({ {0}, {1} });
    inputs.push_back({ {1}, {1} });

    outputs.push_back({ {-1} });
    outputs.push_back({ { 1} });
    outputs.push_back({ { 1} });
    outputs.push_back({ {-1} });

    std::cout << "\033[0;36mDemonstration that the network will not learn the XOR gate using linear activation functions.\n";
    std::cout << "Step 1: learn an XOR gate using linear activation function, " << 10 * STEPS << " steps and 3 hidden layers of 5 neurons each.\n";

    // XOR Gate Train using linear functions
    unsigned int seed = static_cast<unsigned int>(time(0));
    std::cout << "Using seed: " << seed << std::endl << std::endl;
    perceptron_Linear.RandomizeWeights(seed, -1.0f, 1.0f);
    perceptron_Linear.InitTrainCache();

    for (int epoch = 0; epoch < 10 * STEPS; epoch++)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            perceptron_Linear.SetInputValues(inputs[i]);
            perceptron_Linear.ForwardPropagationWithCache(Linear, LinearDerivative);
            perceptron_Linear.BackwardPropagation(outputs[i], LEARNING_RATE_LINEAR, MOMENT);
        }
    }
    perceptron_Linear.ClearTrainCache();

    // XOR Gate Result
    std::cout << "\033[0;32m";
    for (int i = 0; i < inputs.size(); i++)
    {
        perceptron_Linear.SetInputValues(inputs[i]);
        const Matrix<float>& out = perceptron_Linear.ForwardPropagation(Linear);

        int binaryValue = BinaryStep(out(0, 0));
        std::cout << inputs[i](0, 0) << " XOR " << inputs[i](1, 0) << " = " << binaryValue << " (" << out(0, 0) << ")" << std::endl;
    }

    std::cout << "\n\033[0;37mWeights: \n" << perceptron_Linear << std::endl;
    
    std::cout << "\033[0;36mStep 2: learn an XOR gate using non-linear activation function, " << STEPS << " steps and one hidden layers with 2 neurons.\n";

    // XOR Gate Train using non-linear functions
    perceptron_NonLinear.RandomizeWeights(seed, -1.0f, 1.0f);
    perceptron_NonLinear.InitTrainCache();
    for (int epoch = 0; epoch < STEPS; epoch++)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            perceptron_NonLinear.SetInputValues(inputs[i]);
            perceptron_NonLinear.ForwardPropagationWithCache(HyperbolicTangent, HyperbolicTangentDerivativeOptimized, true);
            perceptron_NonLinear.BackwardPropagation(outputs[i], LEARNING_RATE_NONLINEAR, MOMENT);
        }
    }
    perceptron_NonLinear.ClearTrainCache();

    // XOR Gate Result
    std::cout << "\033[0;32m";
    for (int i = 0; i < inputs.size(); i++)
    {
        perceptron_NonLinear.SetInputValues(inputs[i]);
        const Matrix<float>& out = perceptron_NonLinear.ForwardPropagation(HyperbolicTangent);

        int binaryValue = BinaryStep(out(0, 0));
        std::cout << inputs[i](0, 0) << " XOR " << inputs[i](1, 0) << " = " << binaryValue << " (" << out(0, 0)  << ")" << std::endl;
    }

    std::cout << "\n\033[0;37mWeights: \n" << perceptron_NonLinear << std::endl;

    getc(stdin);
    return 0;
}