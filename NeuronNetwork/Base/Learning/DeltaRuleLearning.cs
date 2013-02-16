using System;
using System.Collections.Generic;
using System.Linq;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Layers;
using NeuronNetwork.Base.Neurons;

namespace NeuronNetwork.Base.Learning
{
    public class DeltaRuleLearning : ISupervisedLearning
    {
        private readonly ILayer _layer;
        private double _learningRate = 0.1;

        public DeltaRuleLearning(ILayer layer)
        {
            this._layer = layer;
        }

        /// <summary>
        /// Learning rate
        /// </summary>
        /// 
        /// <remarks>The value determines speed of learning  in the range of [0, 1].
        /// Default value equals to 0.1.</remarks>
        /// 
        public double LearningRate
        {
            get { return this._learningRate; }
            set { this._learningRate = Math.Max(0.0, Math.Min(1.0, value)); }
        }

        #region ILearning Members

        public double Run(double[] input, double[] output)
        {
            IList<double> layerOutput = this._layer.Compute(new List<double>(input));

            IActivationFunction activationFunction = this._layer.Neurons[0].ActivationFunction;

            double error = 0.0;

            for (int j = 0; j < this._layer.Neurons.Count; j++)
            {
                INeuron neuron = this._layer.Neurons[j];

                double e = output[j] - layerOutput[j];
                double functionDerivative = activationFunction.Derivative2(layerOutput[j]);

                for (int i = 0; i < neuron.InputsCount; i++)
                    neuron.Weights[i] += this._learningRate * e * functionDerivative * input[i];

                neuron.Threshold += this._learningRate * e * functionDerivative;

                error += (e * e);
            }

            return error / 2;
        }

        public double RunEpoch(double[][] input, double[][] output)
        {
            return input.Select((t, i) => this.Run(t, output[i])).Sum();
        }

        #endregion
    }
}