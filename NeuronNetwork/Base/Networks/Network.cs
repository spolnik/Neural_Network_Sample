using System;
using System.Collections.Generic;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Layers;


namespace NeuronNetwork.Base.Networks
{
    public class Network : INetwork
    {
        public Network(int inputsCount, IActivationFunction function, List<LayerData> layersData)
        {
            this.InputsCount = Math.Max(1, inputsCount);
            this.Layers = new LayersCollection(inputsCount, function, layersData);
        }

        #region INetwork Members

        public void Randomize(double min, double max)
        {
            this.Layers.Randomize(min, max);
        }
		
        public int InputsCount { get; private set; }
		
	
        public ILayersCollection Layers { get; private set; }

        public IList<double> Outputs { get; private set; }
		
		
        public IList<double> Compute(List<double> inputs)
        {
            this.Outputs = new OutputsCollection(inputs);

            foreach (ILayer layer in this.Layers)
                this.Outputs = new OutputsCollection(layer.Compute(this.Outputs));

            return this.Outputs;
        }

        #endregion
    }
}