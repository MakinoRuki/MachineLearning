#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <random>
using namespace std;
int n, m, k;
double w0;
vector<vector<double> > correlations;
vector<double> weights;
const char* kTestFile = "/Users/ruki/Documents/Developer/MachineLearning/test.csv";
const char* kWeightsFile = "/Users/ruki/Documents/Developer/MachineLearning/model.txt";
bool ReadWeightsFile() {
    freopen(kWeightsFile, "r", stdin);
    cin>>m>>k;
    for (int i = 0; i < m; ++i) {
        vector<double> correlation;
        for (int j = 0; j < k; ++j) {
            double w;
            cin>>w;
            correlation.push_back(w);
        }
        correlations.push_back(correlation);
    }
    for (int i = 0; i < m; ++i) {
        double w;
        cin>>w;
        weights.push_back(w);
    }
    cin>>w0;
    return true;
}
void RunTest() {
    ifstream fin(kTestFile);
    string line;
    vector<double> tests;
    while(getline(fin, line)) {
        istringstream sin(line);
		string feature;
		while (getline(sin, feature, ',')) {
			tests.push_back(stod(feature));
		}
    }
    // for (int i = 0; i < tests.size(); ++i) {
    //     cout<<"test : "<<tests[i]<<endl;
    // }
    if (tests.size() != m) {
        cout<<"Feature size not same with model !!!"<<endl;
        return;
    }
    fin.close();
    double y1 = w0;
    for (int j = 0; j < m; ++j) {
        y1 += weights[j] * tests[j];
    }
    double corweights = 0.0;
    for (int a = 0; a < k; ++a) {
        double w1 = 0.0;
        double w2 = 0.0;
        for (int b = 0; b < m; ++b) {
            w1 += correlations[b][a] * tests[b];
            w2 += correlations[b][a] * correlations[b][a] * tests[b] * tests[b];
        }
        w1 = w1 * w1;
        corweights += w1 - w2;
    }
    corweights *= 0.5;
    y1 += corweights;
    cout<<"User score : "<<y1<<endl;
}
int main() {
    bool status = ReadWeightsFile();
    if (!status) {
        cout<<"Bad model !!!"<<endl;
        return 0;
    }
    RunTest();
    return 0;
}