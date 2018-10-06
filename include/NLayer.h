#pragma once

#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;

float random()
{
	return rand() / float(RAND_MAX);
}

class NLayer
{
public:
	NLayer(unsigned int _size, float *_outputs = nullptr):
		size(_size), outputs(_outputs) {}

	void set(float *_outputs)
	{
		outputs = _outputs;
	}

	inline float output(unsigned int index) const
	{
		return outputs[index % size];
	}

	virtual ~NLayer() {}
public:
	const unsigned int size;
private:
	float *outputs;
};

class FCNLayer : public NLayer
{
public:
	FCNLayer(unsigned int _size, NLayer *_previous) :
		NLayer(_size), nOutputs(new float [_size]), previous(_previous), next(this)
	{
		set(nOutputs);
		lossDerivatives = new float[size];
		nWeights = new float[size * (previous->size + 1)];
		initializeWeights();
	}

	void connect(FCNLayer *_next)
	{
		next = _next;
	}

	FCNLayer* getNext()
	{
		return next;
	}

	NLayer* getPrevious()
	{
		return previous;
	}

	bool isLast() const
	{
		return next == this;
	}

	void updateWeights(float learningRate)
	{
		#pragma omp parallel for
		for (int i = 0; i < size; ++i)
		{
			for (unsigned int j = 0; j < previous->size; ++j)
				weight(j, i) -= derivative(i) * previous->output(j) * learningRate;
			weight(previous->size, i) -= derivative(i) * learningRate;
		}
	}

	void initializeWeights()
	{
		for (unsigned int i = 0; i < size * (previous->size + 1); ++i)
			nWeights[i] = (random() - 0.5f) * (2.f / (previous->size + next->size));
	}

	void updateWSum()
	{
		#pragma omp parallel for
		for (int i = 0; i < size; ++i)
		{
			nOutputs[i] = 0.f;
			for (unsigned int j = 0; j < previous->size; ++j)
				nOutputs[i] += weight(j, i) * previous->output(j);
			nOutputs[i] += weight(previous->size, i);
		}
	}

	virtual void applyActivationFunction()
	{
		#pragma omp parallel for
		for (int i = 0; i < size; ++i)
			nOutputs[i] = tanhf(nOutputs[i]);
	}

	void updateOutputs()
	{
		updateWSum();
		applyActivationFunction();
	}

	void updateTanhDerivatives()
	{
		#pragma omp parallel for
		for (int i = 0; i < size; ++i)
		{
			lossDerivatives[i] = 0.f;
			for (unsigned int j = 0; j < next->size; ++j)
				lossDerivatives[i] += next->derivative(j) * next->weight(i, j);
			lossDerivatives[i] *= 1.f - nOutputs[i] * nOutputs[i];
		}
	}

	inline const float& weight(unsigned int wIndex, unsigned int nIndex) const
	{
		return weight(wIndex, nIndex);
	}

	inline float& weight(unsigned int wIndex, unsigned int nIndex)
	{
		return nWeights[nIndex * (previous->size + 1) + wIndex];
	}

	inline float derivative(unsigned int nIndex) const
	{
		return lossDerivatives[nIndex % size];
	}

	~FCNLayer()
	{
		delete[] nOutputs;
		delete[] nWeights;
		delete[] lossDerivatives;
	}
protected:
	NLayer *previous;
	FCNLayer *next;
	float *nOutputs;
	float *nWeights;
	float *lossDerivatives;
};

class SoftmaxNLayer : public FCNLayer
{
public:
	SoftmaxNLayer(unsigned int _size, NLayer *previous):
		FCNLayer(_size, previous) {}

	void applyActivationFunction() override
	{
		float max = 0.f;
		for (unsigned int i = 0; i < size; ++i)
			if (nOutputs[i] > max)
				max = nOutputs[i];
		float sum = 0.f;

		#pragma omp parallel for reduction(+: sum)
		for (int i = 0; i < size; ++i)
			sum += expf(nOutputs[i] - max);

		sum = 1.f / sum;

		#pragma omp parallel for
		for (int i = 0; i < size; ++i)
			nOutputs[i] *= sum * expf(nOutputs[i] - max);
	}

	void setDerivatives(unsigned int label)
	{
		for (unsigned int i = 0; i < size; ++i)
			lossDerivatives[i] = nOutputs[i];
		lossDerivatives[label % size] -= 1.f;
	}
};