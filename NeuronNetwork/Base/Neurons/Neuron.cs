using System;
using System.Collections.Generic;
using System.Threading;
using NeuronNetwork.Base.Functions;

namespace NeuronNetwork.Base.Neurons
{
    public class Neuron : INeuron
    {
        public Neuron(int inputsCount, IActivationFunction function)
        {
            this.Output = 0.0;
            this.Threshold = 0.0;
            this.InputsCount = Math.Max(1, inputsCount);
            this.Weights = new List<double>();
            this.ActivationFunction = function;
        }

        #region INeuron Members

        public double Threshold { get; set; }

        public double Output { get; private set; }

        public int InputsCount { get; private set; }

        public List<double> Weights { get; set; }

        public IActivationFunction ActivationFunction { get; private set; }

        public double Compute(IList<double> inputs)
        {
			
            if (inputs.Count != this.InputsCount)
            {
                throw new ArgumentException("Inputs count must be equal to InputsCount property of Neuron object!",
                                            "inputs" + inputs.Count + " " + this.InputsCount);
            }

            double initialSumValue = 0.0;

            for (int i = 0; i < this.InputsCount; i++)
            {
                initialSumValue += this.Weights[i] * inputs[i];
				//new approach:
				//initialSumValue += Math.Abs(inputs[i] - this.Weights[i]);
			
			}
			if( this.ActivationFunction is ThresholdFunction)
				((ThresholdFunction)this.ActivationFunction).Threshold = this.Threshold;

			
			this.Output = this.ActivationFunction.Function(initialSumValue + this.Threshold );
			
            return this.Output;
        }

        public void Randomize(double min, double max)
        {
            if (min > max)
            {
                double temp = min;
                min = max;
                max = temp;
            }

            for (int i = 0; i < this.InputsCount; i++)
            {
                Thread.Sleep(2);
                this.Weights.Add(GetRandomValue(min, max - min));
            }

            this.Threshold = GetRandomValue(min, max - min);
        }

        #endregion

        private static double GetRandomValue(double min, double length)
        {
            return new Random((int) DateTime.Now.Ticks).NextDouble() * length + min;
        }
    }
}