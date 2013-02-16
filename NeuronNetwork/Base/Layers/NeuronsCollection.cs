using System;
using System.Collections.Generic;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Neurons;

namespace NeuronNetwork.Base.Layers
{
    public class NeuronsCollection : List<INeuron>, INeuronsCollection
    {
        public NeuronsCollection(int neuronsCount, int inputsCount, IActivationFunction function) : base(neuronsCount)
        {
            this.InitialNeurons(neuronsCount, inputsCount, function);
        }

        #region INeuronsCollection Members

        public void Randomize(double min, double max)
        {
            foreach (INeuron neuron in this)
            {
                neuron.Randomize(min, max);
            }
        }

		public void ApplyThresholds(LayerData layerData){
			for (int i = 0; i < this.Count; i++)
            {
                if (layerData.Thresholds.Count == 0)
                    continue;

                this[i].Threshold = layerData.Thresholds[i];
            }
			
		}
		
        public void ApplyWeights(LayerData layerData)
        {
            for (int i = 0; i < this.Count; i++)
            {
                if (layerData.Weights.Count == 0)
                    continue;

                Console.Write(layerData.Weights.Count + " " + this.Count);
                this[i].Weights = layerData.Weights[i];
            }
        }

        #endregion

        private void InitialNeurons(int count, int inputsCount, IActivationFunction function)
        {
            for (int i = 0; i < count; i++)
                this.Add(new Neuron(inputsCount, function));
        }
    }
}