using System;
using System.Collections.Generic;
using System.Linq;
using NeuronNetwork.Base.Layers;
using NeuronNetwork.Base.Neurons;

namespace NeuronNetwork.Base.Learning
{
    public class PerceptronLearning : ISupervisedLearning
    {
        private readonly ILayer _layer;
        private double _learningRate = 0.1;

        /// <summary>
        /// Learning rate
        /// </summary>
        /// 
        /// <remarks>The value determines speed of learning in the range of [0, 1].
        /// Default value equals to 0.1.</remarks>
        /// 
        public double LearningRate
        {
            get { return this._learningRate; }
            set
            {
                this._learningRate = Math.Max(0.0, Math.Min(1.0, value));
            }
        }

        public PerceptronLearning(ILayer layer)
        {
            this._layer = layer;
        }

        /// <summary>
        /// Runs learning iteration
        /// </summary>
        /// 
        /// <param name="input">input vector</param>
        /// <param name="output">desired output vector</param>
        /// 
        /// <returns>Returns absolute error - difference between real output and
        /// desired output</returns>
        /// 
        /// <remarks>Runs one learning iteration and updates neuron's
        /// weights in case if neuron's output does not equal to the
        /// desired output.</remarks>
        /// 
        public double Run(double[] input, double[] output)
        {
            IList<double > layerOutputs = this._layer.Compute(input);

            double error = 0.0;

            for (int j = 0; j < this._layer.Neurons.Count; j++)
            {
                double e = output[j] - layerOutputs[j];

                if (e == 0) 
                    continue;

                INeuron perceptron = this._layer.Neurons[j];

                for (int i = 0; i < perceptron.InputsCount; i++)
                    perceptron.Weights[i] += this._learningRate * e * input[i];

                perceptron.Threshold += this._learningRate * e;

                error += Math.Abs(e);
            }

            return error;
        }

        /// <summary>
        /// Runs learning epoch
        /// </summary>
        /// 
        /// <param name="input">array of input vectors</param>
        /// <param name="output">array of output vectors</param>
        /// 
        /// <returns>Returns sum of absolute errors</returns>
        /// 
        /// <remarks>Runs series of learning iterations - one iteration
        /// for each input sample. Updates neuron's weights each time,
        /// when neuron's output does not equal to the desired output.</remarks>
        ///
        public double RunEpoch(double[][] input, double[][] output)
        {
            return input.Select((t, i) => this.Run(t, output[i])).Sum();
        }
    }
}