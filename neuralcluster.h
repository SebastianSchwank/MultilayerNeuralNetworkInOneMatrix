#ifndef NEURALCLUSTER_H
#define NEURALCLUSTER_H


#include <vector>
#include <iostream>
#include <cmath>
#include <QDebug>

//Ideas:
//Probailistic Firering
//Dropout
//Momentum

using namespace std;

class NeuralCluster
{
public:
    NeuralCluster(int inputs, int outputs, int hidden);

    void propergate(vector<float> input, vector<float> output, float learningRate);
    vector<vector<float>> getWeights();
    void train();

    vector<float> getActivation();
    vector<float> getCounterActivation();
    vector<float> getError();

    void syncronize();

private:
    int                   numInputs;
    int                   numOutputs;
    vector<vector<float>> weights;
    vector<float>         realNetActivation;
    vector<float>         counterNetActivation;
    vector<float>         lastRealNetActivation;
    vector<float>         lastcounterNetActivation;
    vector<float>         meanRealNetActivation;
    vector<float>         meanCounterNetActivation;
    vector<float>         errorNet;
    vector<float>         realNetActivationDerivative;
    vector<float>         counterNetActivationDerivative;
    vector<float>         lastError;

};

#endif // NEURALCLUSTER_H
