using System;

namespace NeuronNetwork.Base.Functions
{
    public class ThresholdFunction : IActivationFunction
    {
        public ThresholdFunction()
        {
            this.Threshold = 0.2015972;
        }

        public ThresholdFunction(double threshold)
        {
            this.Threshold = threshold;
        }

        public double Threshold { get; set; }

        #region IActivationFunction Members

        public double Function(double x)
        {
            Console.WriteLine("x: " + x);
            return x > this.Threshold ? 1 : 0;
        }

        public double Derivative(double x)
        {
            return 0;
        }

        public double Derivative2(double y)
        {
            return 0;
        }

        #endregion
    }
}