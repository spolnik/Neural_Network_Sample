using System.Collections.Generic;
using NeuronNetwork.Base.Neurons;

namespace NeuronNetwork.Base.Layers
{
    public interface INeuronsCollection : ICollection<INeuron>
    {
        INeuron this[int index] { get; set; }
        void Randomize(double min, double max);
        void ApplyWeights(LayerData layerData);
		void ApplyThresholds(LayerData layerData);
    }
}