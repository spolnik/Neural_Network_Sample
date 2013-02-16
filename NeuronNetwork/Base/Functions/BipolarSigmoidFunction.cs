using System;

namespace NeuronNetwork.Base.Functions
{
    public class BipolarSigmoidFunction : IActivationFunction
    {
        public BipolarSigmoidFunction()
        {
            this.Alpha = 2;
        }

        public BipolarSigmoidFunction(double alpha)
        {
            this.Alpha = alpha;
        }

        public double Alpha { get; set; }

        #region IActivationFunction Members

        public double Function(double x)
        {
            return ((2 / (1 + Math.Exp(-this.Alpha * x))) - 1);
        }

        public double Derivative(double x)
        {
            double y = this.Function(x);

            return (this.Alpha * (1 - y * y) / 2);
        }

        public double Derivative2(double y)
        {
            return (this.Alpha * (1 - y * y) / 2);
        }

        #endregion
    }
}