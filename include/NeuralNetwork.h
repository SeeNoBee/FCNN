#pragma once

#include <NLayer.h>
#include <initializer_list>
#include <math.h>
#include <iostream>

using namespace std;

class FCNN
{
public:
	FCNN(unsigned int inputSize, unsigned int *innerSizes, unsigned int _hideCount, unsigned int outputSize):
		hide(nullptr), hideCount(_hideCount)
	{
		initialize(inputSize, innerSizes, outputSize);
	}

	FCNN(unsigned int inputSize, std::initializer_list<unsigned int> innerSizes, unsigned int outputSize):
		hide(nullptr), hideCount(innerSizes.size())
	{
		unsigned int *tmp = nullptr;
		if (innerSizes.size() > 0)
		{
			tmp = new unsigned int[innerSizes.size()];
			unsigned int i = 0;
			for (auto size : innerSizes)
				tmp[i++] = size;
		}

		initialize(inputSize, tmp, outputSize);

		delete[] tmp;
	}

	void recognize()
	{
		for (unsigned int i = 0; i <= hideCount; ++i)
			hide[i]->updateOutputs();
	}

	unsigned int result()
	{
		float max = output->output(0);
		unsigned int label = 0;

		for (unsigned int i = 1; i < output->size; ++i)
			if (output->output(i) > max)
			{
				max = output->output(i);
				label = i;
			}

		return label;
	}

	float loss(unsigned int label)
	{
		return -logf(output->output(label));
	}

	void resetWeigth()
	{
		for (unsigned int i = 0; i <= hideCount; ++i)
			hide[i]->initializeWeights();
	}

	void teach(unsigned int label, float learningRate)
	{
		recognize();
		output->setDerivatives(label);
		output->updateWeights(learningRate);
		for (int i = hideCount - 1; i >= 0; --i)
		{
			hide[i]->updateTanhDerivatives();
			hide[i]->updateWeights(learningRate);
		}
	}

	void printWeights()
	{
		for (unsigned int i = 0; i <= hideCount; ++i)
		{
			cout << "layer: " << i << endl;
			for (unsigned int j = 0; j < 10/*hide[i]->size*/; ++j)
			{
				cout << " neuron: " << j << endl;
				for (unsigned int k = 0; k < 10/*hide[i]->getPrevious()->size + 1*/; ++k)
					cout << "  " << hide[i]->weight(k, i) << endl;
			}
		}
	}

	~FCNN()
	{
		delete input;
		for (unsigned int i = 0; i <= hideCount; ++i)
			delete hide[i];
		delete[] hide;
	}
public:
	NLayer *input;
	SoftmaxNLayer *output;
private:
	FCNLayer **hide;
	unsigned int hideCount;

	void initialize(unsigned int inputSize, unsigned int *innerSizes, unsigned int outputSize)
	{
		input = new NLayer(inputSize);
		NLayer *current = input;

		hide = new FCNLayer*[hideCount + 1];
		if (hideCount > 0)
			for (unsigned int i = 0; i < hideCount; ++i)
			{
				hide[i] = new FCNLayer(innerSizes[i], current);
				current = hide[i];
			}

		output = new SoftmaxNLayer(outputSize, current);
		hide[hideCount] = output;

		for (unsigned int i = 0; i < hideCount; ++i)
			hide[i]->connect(hide[i + 1]);
	}
};