using System.Collections.Generic;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Layers;
using NeuronNetwork.Base.Neurons;
using NUnit.Framework;

namespace TestNeuronNetwork
{
    [TestFixture]
    public class TestLayer
    {
        private readonly List<double> _inputs = new List<double> { 0.34, 0.243, 0.111 };
        private const int InputsCount = 3;
        private const int NeuronsCount = 4;
        private ILayer _layer;

        [SetUp]
        public void SetUp()
        {
            this._layer = new Layer(NeuronsCount, InputsCount, new SigmoidFunction());
        }

        [Test]
        public void TestAfterInit()
        {
            Assert.AreEqual(NeuronsCount, this._layer.Neurons.Count);
            Assert.AreEqual(NeuronsCount, this._layer.Outputs.Count);
            Assert.AreEqual(InputsCount, this._layer.InputsCount);

            foreach (INeuron neuron in this._layer.Neurons)
            {
                foreach (double weight in neuron.Weights)
                {
                    Assert.AreEqual(0.0, weight);    
                }
                
                Assert.AreEqual(InputsCount, neuron.InputsCount);
                Assert.AreEqual(0.0, neuron.Output);
            }

            foreach (double output in this._layer.Outputs)
            {
                Assert.AreEqual(0.0, output);
            }
        }

        [Test]
        public void TestRandomizeNeuronsWeights()
        {
            this._layer.Neurons.Randomize(0.0, 1.0);

            foreach (INeuron neuron in this._layer.Neurons)
            {
                foreach (double weight in neuron.Weights)
                {
                    Assert.GreaterOrEqual(weight, 0.0);
                    Assert.LessOrEqual(weight, 1.0);
                }
            }
        }

        [Test]
        public void TestCompute()
        {
            this._layer.Neurons.Randomize(0.0, 1.0);

            this._layer.Compute(this._inputs);

            foreach (double output in this._layer.Outputs)
            {
                Assert.AreNotEqual(0.0, output);
            }
        }
    }
}