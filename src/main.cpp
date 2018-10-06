#include <NeuralNetwork.h>
#define USE_MNIST_LOADER
#define MNIST_FLOAT
#include <MNIST.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>

using namespace std;

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		cout << "FCNN launch format: [learning_rate] [era_count] [hidden_layer1_size hidden_layer2_size ...]" << endl;
		return 1;
	}

	std::ios::sync_with_stdio(false);
	srand(time(0));

	mnist_data *train;
	mnist_data *test;
	unsigned int cntTrain;
	unsigned int cntTest;

	mnist_load("train-images.idx3-ubyte", "train-labels.idx1-ubyte", &train, &cntTrain);
	mnist_load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", &test, &cntTest);

	float learningRate = atof(argv[1]);
	unsigned int eraCount = atoi(argv[2]);

	unsigned int *sizes = nullptr;
	unsigned int hiddenCount = argc - 3;
	if (hiddenCount > 0)
	{
		sizes = new unsigned int[hiddenCount];

		for (unsigned int i = 0; i < hiddenCount; ++i)
			sizes[i] = atoi(argv[i + 3]);
	}

	vector<unsigned int> indexes;
	indexes.resize(cntTrain);
	for (unsigned int i = 0; i < cntTrain; ++i)
		indexes[i] = i;

	FCNN net(28 * 28, sizes, hiddenCount, 10);

	for (unsigned int era = 0; era < eraCount; ++era)
	{
		//cout << "era: " << era << endl;
		random_shuffle(indexes.begin(), indexes.end());
		for (unsigned int i = 0; i < cntTrain; ++i)
		{
			net.input->set(train[indexes[i]].data);
			net.teach(train[indexes[i]].label, learningRate);
			if (i % 100 == 0)
				cout <<"\rera: " << era << " progress: " << (i / double(cntTrain)) * 100 << "%         ";
		}
	}

	unsigned int accuracy = 0;
	for (unsigned int i = 0; i < cntTest; ++i)
	{
		net.input->set(test[i].data);
		net.recognize();
		if (net.result() == test[i].label)
			++accuracy;
	}

	cout << "\raccuracy: " << (double(accuracy) / cntTest) * 100 << "%          " << endl;

	delete[] sizes;

    return 0;
}