#ifndef FEEDFORWARDNETWORKBP_H
#define FEEDFORWARDNETWORKBP_H

#include <vector>

using namespace std;

class feedForwardNetworkBP
{
public:
    feedForwardNetworkBP();

    vector<float> feedForward(vector<float> inputActivation);
    void train(vector<float> error);

    vector<vector<vector<float>>> getLayers();

private:
    vector<vector<float>> weightsL0;
    vector<vector<float>> weightsL1;
};

#endif // FEEDFORWARDNETWORKBP_H
