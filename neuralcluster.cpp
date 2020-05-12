#include "neuralcluster.h"


NeuralCluster::NeuralCluster(int inputs, int outputs, int hidden){

    vector<vector<float> > weightsCreation;
    for(int i = 0; i < inputs+outputs+hidden; i++){
        meanRealNetActivation.push_back(0.0);
        meanCounterNetActivation.push_back(0.0);
        realNetActivation.push_back(0.0);
        counterNetActivation.push_back(0.0);
        realNetActivationDerivative.push_back(0.0);
        counterNetActivationDerivative.push_back(0.0);
        errorNet.push_back(0.0);
        lastError.push_back(0.0);
        vector<float> weightColumn;
        for(int j = 0; j < inputs+outputs+hidden; j++){
            weightColumn.push_back(5.0*(1.0*rand()/RAND_MAX-0.5));
        }
        weights.push_back(weightColumn);
    }

    numInputs = inputs;
    numOutputs = outputs;

    lastRealNetActivation = realNetActivation;
    lastcounterNetActivation = counterNetActivation;


}

vector<vector<float>> NeuralCluster::getWeights(){
    return weights;
}

void NeuralCluster::train(){
    vector<float> errorVector;
    for(int i = 0; i < weights.size(); i++) errorVector.push_back(counterNetActivation[i]-realNetActivation[i]);
    for(int i = 0; i < weights.size(); i++){
        for(int j = 0; j < weights[i].size(); j++){

            if((i >= 0)&& (j >= 0) && (i <= numInputs)&& (j <= numInputs)) weights[i][j] = 0.0;
            if((i >= numInputs)&& (j >= 0) && (i <= numInputs+numOutputs)&& (j <= numInputs+numOutputs)) weights[i][j] = 0.0;
            if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (i <= weights.size()-1)&& (j <= weights.size()-1)) weights[i][j] = 0.0;

            if(weights[i][j]!=0.0){
                if(abs(lastError[i]) < abs(errorVector[i])) weights[i][j] += (realNetActivation[j])*errorVector[i]*0.1;
            }

        }
    }
    lastError = errorVector;
}

vector<float> NeuralCluster::getCounterActivation(){
    return counterNetActivation;
}

vector<float> NeuralCluster::getError(){
    vector<float> error;
    for(int i = 0; i < weights.size(); i++) error.push_back(realNetActivation[i]-counterNetActivation[i]);
    return error;
}

vector<float> NeuralCluster::getActivation(){
    return realNetActivation;
}

void NeuralCluster::syncronize(){
    for (int i = 0; i < weights.size()-1; i++)realNetActivation[i] = counterNetActivation[i];
}

void NeuralCluster::propergate(vector<float> input,vector<float> output, float learningRate){

    for(int i = 0; i < input.size(); i++){ counterNetActivation[i] = input[i]; realNetActivation[i] = input[i]; }
    for(int i = input.size(); i < output.size()+input.size(); i++) { counterNetActivation[i] = output[i-input.size()];}

    vector<float> interimReal = realNetActivation;
    vector<float> interimCounter = counterNetActivation;

    vector<float> nextCounter;
    vector<float> nextReal;

        vector<float> deltaEnergys = counterNetActivation;
        float absDeltaEnergy = 0.0;
        for(int i = 0; i < weights.size(); i++){
            float x = 0.0;
            float y = 0.0;
            float absEnergy;
            float maxActivation = 0.0;
            float minActivation = 0.0;
            for(int j = 0; j < weights[i].size(); j++){
                    float deltaEnergy =  weights[i][j]*(interimCounter[j]);
                    x += deltaEnergy;
            }

            float signum = 0.0;
            if(x > 0.0) signum = 1.0;
            else signum = -1.0;
            deltaEnergys[i] = 1.0/(1.0+exp(-x));
            //cout << deltaEnergys[i];
            //realNetActivationDerivative[i] = cos(x);
        }


        for(int i = input.size()+output.size(); i < weights.size(); i++){
            counterNetActivation[i] = deltaEnergys[i];
        }

        vector<float> deltaEnergysI = realNetActivation;
        float absDeltaEnergyI = 0.0;
        for(int i = 0; i < weights.size(); i++){
            float x = 0.0;
            float y = 0.0;
            float absEnergy;
            float maxActivation = 0.0;
            float minActivation = 0.0;
            for(int j = 0; j < weights[i].size(); j++){
                    float deltaEnergy =  weights[i][j]*(interimReal[j]);
                    x += deltaEnergy;
            }

            //deltaEnergysI[i] = (1.0/(1.0+exp(-x)));
            float signum = 0.0;
            if(x > 0.0) signum = 1.0;
            else signum = -1.0;
            deltaEnergysI[i] = 1.0/(1.0+exp(-x));
            //counterNetActivationDerivative[i] = cos(x);
        }


        for(int i = input.size(); i < weights.size(); i++){
            realNetActivation[i] = deltaEnergysI[i];
        }


        for(int i = 0; i < input.size(); i++){ counterNetActivation[i] = input[i]; realNetActivation[i] = input[i]; }
        for(int i = input.size(); i < output.size()+input.size(); i++) { counterNetActivation[i] = output[i-input.size()];}

}
