namespace NeuronNetwork.Base.Functions
{
    public class LinearFunction : IActivationFunction
    {
        #region IActivationFunction Members

        public double Function(double x)
        {
            return x;
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