#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <random>
#include <cmath>
using namespace std;
vector<vector<double> > samples;
int n, m, k;
double alpha = 1.0;
double w0;
vector<vector<double> > correlations;
vector<double> weights;
const char* kSampleFile = "/Users/ruki/Documents/Developer/MachineLearning/samples2.csv";
const char* kWeightsFile = "/Users/ruki/Documents/Developer/MachineLearning/model.txt";
bool ReadSamples() {
    ifstream fin(kSampleFile);
	string line; 
	while (getline(fin, line)) {
		istringstream sin(line);
		string feature;
        vector<double> sample;
		while (getline(sin, feature, ',')) {
			sample.push_back(stod(feature));
		}
        samples.push_back(sample);
	}
    n = samples.size();
    if (!n) {
        cout<<"No Samples !!"<<endl;
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (samples[i].size() != samples[0].size()) {
            cout<<"Different sample dimensions !!!"<<endl;
            return false;
        }
    }
    m = samples[0].size() - 1;
    // for (int i = 0; i < samples.size(); ++i) {
    //     for (int j = 0; j < samples[i].size(); ++j) {
    //         cout<<samples[i][j]<<" ";
    //     }
    //     cout<<endl;
    // }
    return true;
}
void InitWeights() {
    k = 5;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> range(0.001, 0.005);

    for (int i = 0; i < m; ++i) {
        weights.push_back(range(gen));
        vector<double> v(k, 0.0);
        for (int j = 0; j < k; ++j) {
            v[j] = range(gen);
        }
        correlations.push_back(v);
    }
    w0 = range(gen);
    // cout<<m<<" "<<k<<" "<<n<<endl;
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < k; ++j) {
    //         cout<<correlations[i][j]<<",";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;
    // for (int i= 0; i < m; ++i) {
    //     cout<<weights[i]<<",";
    // }
    // cout<<endl;
    // cout<<w0<<endl;
    // for (int i = 0; i < m; ++i) {
    //     weights.push_back(1);
    //     vector<double> v(k, 1.0);
    //     correlations.push_back(v);
    // }
    // w0 = 1.0;
}
void Train() {
    for (int i = 0; i < n; ++i) {
        double y1 = w0;
        for (int j = 0; j < m; ++j) {
            y1 += weights[j] * samples[i][j];
        }
        double corweights = 0.0;
        for (int a = 0; a < k; ++a) {
            double w1 = 0.0;
            double w2 = 0.0;
            for (int b = 0; b < m; ++b) {
                w1 += correlations[b][a] * samples[i][b];
                w2 += correlations[b][a] * correlations[b][a] * samples[i][b] * samples[i][b];
            }
            w1 = w1 * w1;
            corweights += w1 - w2;
        }
        corweights *= 0.5;
        y1 += corweights;
        double y1s = 5.0 / (1.0 + exp(-y1));
        double y0 = samples[i][m];
      //  cout<<"y1="<<y1<<endl;
        double sigd = exp(-y1) / ((1 + exp(-y1)) * (1 + exp(-y1)));
        sigd *= 5.0;
        w0 -= alpha * (y1s - y0) * sigd;
        for (int j = 0; j < m; ++j) {
            weights[j] -= alpha * (y1s - y0) * sigd * samples[i][j];
            if (i == n-1)cout<<"j="<<weights[j]<<endl;
        }
        for (int a = 0; a < k; ++a) {
            double gradient = 0.0;
            for (int b = 0; b < m; ++b) {
                gradient += correlations[b][a] * samples[i][b];
            }
            for (int b = 0; b < m; ++b) {
                correlations[b][a] -= alpha * (y1s - y0) * sigd * (samples[i][b] * gradient - correlations[b][a] * samples[i][b] * samples[i][b]);
                //cout<<"cab : "<<correlations[b][a]<<endl;
            }
        }
    }
}
void WriteModelToFile() {
	ofstream modelFile;
	modelFile.open(kWeightsFile); // 打开模式可省略
    modelFile << m << " " << k << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            modelFile << correlations[i][j];
            if (j < k - 1) modelFile << " ";
        }
        modelFile << endl;
    }
    for (int i = 0; i < m; ++i) {
        modelFile << weights[i];
        if (i < m - 1) modelFile << " ";
    }
    modelFile << endl;
    modelFile << w0 << endl;
    modelFile.close();
}
int main() {
    bool status = ReadSamples();
    if (!status) {
        cout<<"Bad samples !!!"<<endl;
        return 0;
    }
    InitWeights();
    Train();
    WriteModelToFile();
    return 0;
}