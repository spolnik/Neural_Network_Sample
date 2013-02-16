using System.Collections.Generic;

namespace NeuronNetwork.Base.Networks
{
    public interface INetwork
    {
        int InputsCount { get; }
        ILayersCollection Layers { get; }
        IList<double> Outputs { get; }
        IList<double> Compute(List<double> inputs);
        void Randomize(double min, double max);
    }
}