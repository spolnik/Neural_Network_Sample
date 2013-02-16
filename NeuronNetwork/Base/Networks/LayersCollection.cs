using System.Collections.Generic;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Layers;

namespace NeuronNetwork.Base.Networks
{
    public class LayersCollection : List<ILayer>, ILayersCollection
    {
        public LayersCollection(int inputsCount, IActivationFunction function, List<LayerData> layersData)
        {
            this.InitialLayers(layersData, inputsCount, function);
        }

        #region ILayersCollection Members

        public void Randomize(double min, double max)
        {
            foreach (ILayer layer in this)
                layer.Neurons.Randomize(min, max);
        }

        #endregion

        private void InitialLayers(IList<LayerData> layersData, int inputsCount, IActivationFunction function)
        {
            for (int i = 0; i < layersData.Count; i++)
            {
                ILayer layer = new Layer(layersData[i].CountNeurons, 
                                        (i > 0) ? layersData[i - 1].CountNeurons : inputsCount, 
                                        function);

                if (layersData[i].Weights.Count > 0)
                    layer.Neurons.ApplyWeights(layersData[i]);
                else
                    layer.Neurons.Randomize(0.0, 1.0);

                if (layersData[i].Thresholds.Count > 0)
                    layer.Neurons.ApplyThresholds(layersData[i]);

                this.Add(layer);
            }
        }
    }
}