using System.Collections.Generic;
using NeuronNetwork.Base.Layers;

namespace NeuronNetwork.Base.Networks
{
    public interface ILayersCollection : ICollection<ILayer>
    {
        ILayer this[int index] { get; set; }
        void Randomize(double min, double max);
    }
}