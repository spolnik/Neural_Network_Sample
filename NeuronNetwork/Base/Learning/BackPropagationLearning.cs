using System;
using System.Collections.Generic;
using System.Linq;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Layers;
using NeuronNetwork.Base.Networks;
using NeuronNetwork.Base.Neurons;

namespace NeuronNetwork.Base.Learning
{
    public class BackPropagationLearning : ISupervisedLearning
    {
        private readonly INetwork _network;
		private double _learningRate = 0.1;
		
        private double _momentum = 0.4;

		private readonly double[][]	_neuronErrors;
		private readonly double[][][] _weightsUpdates;
		private readonly double[][] _thresholdsUpdates;
		
		public double LearningRate
		{
			get { return this._learningRate; }
			set
			{
				this._learningRate = Math.Max( 0.0, Math.Min( 1.0, value ) );
			}
		}


		public double Momentum
		{
			get { return this._momentum; }
			set
			{
				this._momentum = Math.Max( 0.0, Math.Min( 1.0, value ) );
			}
		}

		public BackPropagationLearning(INetwork network)
		{
			this._network = network;

			this._neuronErrors = new double[network.Layers.Count][];
			this._weightsUpdates = new double[network.Layers.Count][][];
			this._thresholdsUpdates = new double[network.Layers.Count][];

			for ( int i = 0; i < network.Layers.Count; i++ )
			{
				ILayer layer = network.Layers[i];

				this._neuronErrors[i] = new double[layer.Neurons.Count];
				this._weightsUpdates[i] = new double[layer.Neurons.Count][];
				this._thresholdsUpdates[i] = new double[layer.Neurons.Count];

			    for (int j = 0; j < layer.Neurons.Count; j++)
			        this._weightsUpdates[i][j] = new double[layer.InputsCount];
			}
		}

		public double Run( double[] input, double[] output )
		{
			this._network.Compute(new List<double>(input));

			double error = this.CalculateError( output );

			this.CalculateUpdates(input);

			this.UpdateNetwork( );

			return error;
		}

		public double RunEpoch( double[][] input, double[][] output )
		{
		    return input.Select((t, i) => this.Run(t, output[i])).Sum();
		}

		private double CalculateError( double[] desiredOutput )
		{
			ILayer layerNext;
		    double error = 0;

		    int layersCount = this._network.Layers.Count;

			// assume, that all neurons of the network have the same activation function
			IActivationFunction	function = this._network.Layers[0].Neurons[0].ActivationFunction;

            // first layer
			ILayer layer = this._network.Layers[layersCount - 1];
			double[] errors = this._neuronErrors[layersCount - 1];

			for ( int i = 0; i < layer.Neurons.Count; i++ )
			{
				double output = layer.Neurons[i].Output;
				double e = desiredOutput[i] - output;
				
				errors[i] = e * function.Derivative2( output );
				
				error += ( e * e );
			}

            // other layers
			for ( int j = layersCount - 2; j >= 0; j-- )
			{
				layer		= this._network.Layers[j];
				layerNext	= this._network.Layers[j + 1];
				errors		= this._neuronErrors[j];
				double[] errorsNext = this._neuronErrors[j + 1];

				for ( int i = 0; i < layer.Neurons.Count; i++ )
				{
					double sum = 0.0;
					// for all neurons of the next layer
				    for (int k = 0; k < layerNext.Neurons.Count; k++)
				        sum += errorsNext[k] * layerNext.Neurons[k].Weights[i];

				    errors[i] = sum * function.Derivative2( layer.Neurons[i].Output );
				}
			}

			return error / 2.0;
		}

		private void CalculateUpdates(double[] input)
		{
			INeuron	currentNeuron;
			ILayer layerPrev;

		    double[] neuronWeightUpdates;
			double error;

			// layer fisrt
			ILayer layer = this._network.Layers[0];
			double[] errors = this._neuronErrors[0];
			double[][] layerWeightsUpdates = this._weightsUpdates[0];
			double[] layerThresholdUpdates = this._thresholdsUpdates[0];

			for ( int i = 0; i < layer.Neurons.Count; i++ )
			{
				currentNeuron = layer.Neurons[i];
				error = errors[i];
				neuronWeightUpdates	= layerWeightsUpdates[i];

				for ( int j = 0; j < currentNeuron.InputsCount; j++ )
				{
				    neuronWeightUpdates[j] = this._learningRate * (
				                                                      this._momentum * neuronWeightUpdates[j] +
				                                                      (1.0 - this._momentum) * error * input[j]
				                                                  );
				}

			    layerThresholdUpdates[i] = this._learningRate * (
			                                                        this._momentum * layerThresholdUpdates[i] +
			                                                        (1.0 - this._momentum) * error
			                                                    );
			}

			// other layers
			for ( int k = 1; k < this._network.Layers.Count; k++ )
			{
				layerPrev			= this._network.Layers[k - 1];
				layer				= this._network.Layers[k];
				errors				= this._neuronErrors[k];
				layerWeightsUpdates	= this._weightsUpdates[k];
				layerThresholdUpdates = this._thresholdsUpdates[k];

				for ( int i = 0; i < layer.Neurons.Count; i++ )
				{
					currentNeuron	= layer.Neurons[i];
					error	= errors[i];
					neuronWeightUpdates	= layerWeightsUpdates[i];

					for ( int j = 0; j < currentNeuron.InputsCount; j++ )
					{
					    neuronWeightUpdates[j] = this._learningRate * (
					                                                      this._momentum * neuronWeightUpdates[j] +
					                                                      (1.0 - this._momentum) * error * layerPrev.Neurons[j].Output
					                                                  );
					}

				    layerThresholdUpdates[i] = this._learningRate * (
				                                                        this._momentum * layerThresholdUpdates[i] +
				                                                        (1.0 - this._momentum) * error
				                                                    );
				}
			}
		}

		private void UpdateNetwork()
		{
		    INeuron currentNeuron;
		    ILayer currentLayer;

		    for (int i = 0; i < this._network.Layers.Count; i++)
		    {
		        currentLayer = this._network.Layers[i];
		        double[][] layerWeightsUpdates = this._weightsUpdates[i];
		        double[] layerThresholdUpdates = this._thresholdsUpdates[i];

		        for (int j = 0; j < currentLayer.Neurons.Count; j++)
		        {
		            currentNeuron = currentLayer.Neurons[j];
		            double[] neuronWeightUpdates = layerWeightsUpdates[j];

		            for (int k = 0; k < currentNeuron.InputsCount; k++)
		                currentNeuron.Weights[k] += neuronWeightUpdates[k];

		            currentNeuron.Threshold += layerThresholdUpdates[j];
		        }
		    }
		}
    }
}