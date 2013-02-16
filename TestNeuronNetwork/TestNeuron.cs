using System.Collections.Generic;
using NeuronNetwork.Base.Functions;
using NeuronNetwork.Base.Neurons;
using NUnit.Framework;

namespace TestNeuronNetwork
{
    [TestFixture]
    public class TestNeuron
    {
        private const int DefaultInputsCount = 3;
        private INeuron _neuron;
        private readonly List<double> _inputs = new List<double> {0.34, 0.243, 0.111};

        [SetUp]
        public void SetUp()
        {
            this._neuron = new Neuron(DefaultInputsCount, new SigmoidFunction());
        }

        [Test]
        public void TestAfterInit()
        {
            Assert.AreEqual(DefaultInputsCount, this._neuron.InputsCount);
            Assert.AreEqual(0.0, this._neuron.Output);
            Assert.AreEqual(0.0, this._neuron.Threshold);

            foreach (double weight in this._neuron.Weights)
            {
                Assert.AreEqual(0.0, weight);    
            }
        }

        [Test]
        public void TestRandomizeWeights()
        {
            this._neuron.Randomize(0.0, 1.0);

            foreach (double weight in this._neuron.Weights)
            {
                Assert.GreaterOrEqual(weight, 0.0);
                Assert.LessOrEqual(weight, 1.0);
            }

            Assert.GreaterOrEqual(this._neuron.Threshold, 0.0);
            Assert.LessOrEqual(this._neuron.Threshold, 1.0);
        }

        [Test]
        public void TestCompute()
        {
            this._neuron.Randomize(0.0, 1.0);

            this._neuron.Compute(this._inputs);

            Assert.AreNotEqual(0.0, this._neuron.Output);
        }
    }
}