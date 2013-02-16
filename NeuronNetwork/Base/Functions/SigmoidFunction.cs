using System;

namespace NeuronNetwork.Base.Functions
{
    public class SigmoidFunction : IActivationFunction
    {
        public SigmoidFunction()
        {
            this.Alpha = 2;
        }

        public SigmoidFunction(double alpha)
        {
            this.Alpha = alpha;
        }

        public double Alpha { get; set; }

        #region IActivationFunction Members

        public double Function(double x)
        {
            return (1 / (1 + Math.Exp(-this.Alpha * x)));
        }

        public double Derivative(double x)
        {
            double y = Function(x);

            return (this.Alpha * y * (1 - y));
        }

        public double Derivative2(double y)
        {
            return (this.Alpha * y * (1 - y));
        }

        #endregion
    }
}