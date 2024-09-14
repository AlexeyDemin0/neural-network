//
// Example 1: AND and OR logical gates
//

#include <iostream>
#include <cstdio>
#include <vector>
#include <ctime>

#include "perceptron.h"
#include "math/matrix.h"
#include "math/functions.h"

#define STEPS 100
#define LEARNING_RATE 0.1f
#define MOMENT 0.9f

using namespace NeuralNetwork;
using namespace NeuralNetwork::Math;
using namespace NeuralNetwork::Math::Functions;

int main()
{
    std::vector<int> neuronsPerLayer = { 2, 1 };
    Perceptron<float> perceptron(neuronsPerLayer);

    std::vector<Matrix<float>> inputs;
    std::vector<Matrix<float>> outputs_OR;
    std::vector<Matrix<float>> outputs_AND;

    inputs.push_back({ {0}, {0} });
    inputs.push_back({ {1}, {0} });
    inputs.push_back({ {0}, {1} });
    inputs.push_back({ {1}, {1} });

    outputs_OR.push_back({ {-1} });
    outputs_OR.push_back({ { 1} });
    outputs_OR.push_back({ { 1} });
    outputs_OR.push_back({ { 1} });

    outputs_AND.push_back({ {-1} });
    outputs_AND.push_back({ {-1} });
    outputs_AND.push_back({ {-1} });
    outputs_AND.push_back({ { 1} });

    // AND Gate Train
    unsigned int seed = static_cast<unsigned int>(time(0));
    std::cout << "\033[0;36mUsing seed: " << seed << std::endl << std::endl;
    perceptron.RandomizeWeights(seed, -1.0f, 1.0f);
    perceptron.InitTrainCache();

    for (int epoch = 0; epoch < STEPS; epoch++)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            perceptron.SetInputValues(inputs[i]);
            perceptron.ForwardPropagationWithCache(Linear, LinearDerivative);
            perceptron.BackwardPropagation(outputs_AND[i], LEARNING_RATE, MOMENT);
        }
    }
    
    // AND Gate Result
    std::cout << "\033[0;32m";
    for (int i = 0; i < inputs.size(); i++)
    {
        perceptron.SetInputValues(inputs[i]);
        const Matrix<float>& out = perceptron.ForwardPropagation(Linear);

        float binaryValue = BinaryStep(out(0, 0));
        std::cout << inputs[i](0, 0) << " AND " << inputs[i](1, 0) << " = " << binaryValue << " (" << out(0, 0) << ")" << std::endl;
    }

    std::cout << "\n\033[0;37mWeights: \n" << perceptron << std::endl;

    // OR Gate Train
    perceptron.RandomizeWeights(time(0), -1.0f, 1.0f);
    perceptron.InitTrainCache();

    for (int epoch = 0; epoch < STEPS; epoch++)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            perceptron.SetInputValues(inputs[i]);
            perceptron.ForwardPropagationWithCache(Linear, LinearDerivative);
            perceptron.BackwardPropagation(outputs_OR[i], LEARNING_RATE, MOMENT);
        }
    }

    perceptron.ClearTrainCache();

    // OR Gate Result
    std::cout << "\033[0;32m";
    for (int i = 0; i < inputs.size(); i++)
    {
        perceptron.SetInputValues(inputs[i]);
        const Matrix<float>& out = perceptron.ForwardPropagation(Linear);

        float binaryValue = BinaryStep(out(0, 0));
        std::cout << inputs[i](0, 0) << " OR " << inputs[i](1, 0) << " = " << binaryValue << " (" << out(0, 0) << ")" << std::endl;
    }

    std::cout << "\n\033[0;37mWeights: \n" << perceptron << std::endl;

    getc(stdin);
    return 0;
}