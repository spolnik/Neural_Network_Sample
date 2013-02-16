using System.Collections.Generic;
using NeuronNetwork.Base;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Networks;

namespace NeuronNetwork.Utility
{
    public class NeuralNetworkFileParser
    {
        private const string SigmoidFunctionName = "sigmoid";
		private const string LinearFunctionName = "linear";
		private const string ThresholdFunctionName = "threshold";
		
        private int _inputsCount;
        private IActivationFunction _function;
        private List<LayerData> _layersData;

        public NeuralNetworkFileParser()
        {
            this._inputsCount = 0;
            this._function = null;
            this._layersData = null;
        }

        public INetwork CreateNetwork(string neuronsFilePath, string functionName, int inputsCount)
        {
            this.EvalFunction(functionName);
            this.EvalNeurons(neuronsFilePath);

            this._inputsCount = inputsCount;

            if (this._inputsCount < 1 || this._function == null || this._layersData == null)
            {
                return null;
            }

            INetwork network = new Network(this._inputsCount, this._function, this._layersData);

            return network;
        }

        private void EvalNeurons(string neuronsFilePath)
        {
            DataLoader.Load(neuronsFilePath, out this._layersData);
        }

        private void EvalFunction(string functionName)
        {
            switch (functionName.ToLower())
            {
                case SigmoidFunctionName:
                    this._function = new SigmoidFunction();
                    break;
				case LinearFunctionName:
                    this._function = new LinearFunction();
                    break;
				case ThresholdFunctionName:
                    this._function = new ThresholdFunction();
                    break;
                default:
                    this._function = null;
                    break;
            }
        }
    }
}