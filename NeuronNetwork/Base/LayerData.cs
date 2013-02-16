using System.Collections.Generic;

namespace NeuronNetwork.Base
{
    public class LayerData
    {
        private readonly List<List<double>> _weights;
		public List<double> Thresholds {get; set;}

        public LayerData()
        {
            this._weights = new List<List<double>>();
			this.Thresholds = new List<double>();
        }

        public int CountNeurons { get; set; }

        public List<List<double>> Weights
        {
            get { return this._weights; }
        }
    }
}