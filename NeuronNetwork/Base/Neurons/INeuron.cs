using System.Collections.Generic;
using NeuronNetwork.Base.Functions;

namespace NeuronNetwork.Base.Neurons
{
    public interface INeuron
    {
        double Output { get; }
        double Threshold { get; set; }
        int InputsCount { get; }
        List<double> Weights { get; set; }
        IActivationFunction ActivationFunction { get; }
        double Compute(IList<double> input);
        void Randomize(double min, double max);
    }
}