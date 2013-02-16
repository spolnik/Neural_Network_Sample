using System;
using System.IO;
using NeuronApplication.Properties;
using NeuronNetwork.Base.Networks;
using NeuronNetwork.Utility;

namespace NeuronApplication
{
    public class Program
    {
        private readonly INetwork _network;

        private readonly NeuralNetworkFileParser _parser;

        public Program()
        {
            this._parser = new NeuralNetworkFileParser();
            this._network = this._parser.CreateNetwork(Settings.Default.NeuronsFilePath, Settings.Default.Function, Settings.Default.InputsCount);
        }

        public bool IsCorrectlyInitialized()
        {
            return this._network != null;
        }

        public void Start()
        {
            this._network.Compute(DataLoader.ReadInputs(Settings.Default.InputsFilePath));
        }

        static void Main()
        {
            ValidConfigurationFile();

            Program program = new Program();

            if (!program.IsCorrectlyInitialized())
            {
                throw new ApplicationException("Application is not correctly initialized from the init files.");
            }

            program.Start();

            Console.WriteLine("Output:");
            program.WriteOutput();
        }

        private void WriteOutput()
        {
            foreach (double output in this._network.Outputs)
            {
                Console.WriteLine(output);
            }
        }

        private static void ValidConfigurationFile()
        {
            if (String.IsNullOrEmpty(Settings.Default.NeuronsFilePath))
            {
                throw new ApplicationException("You must define NeuronsFilePath in configuration file!.");
            }

            if (String.IsNullOrEmpty(Settings.Default.InputsFilePath))
            {
                throw new ApplicationException("You must define WeightFilePath in configuration file!.");
            }

            if (!File.Exists(Settings.Default.NeuronsFilePath))
            {
                throw new ApplicationException(string.Format("You must create {0} file!.", Settings.Default.NeuronsFilePath));
            }

            if (!File.Exists(Settings.Default.InputsFilePath))
            {
                throw new ApplicationException(string.Format("You must create {0} file!.", Settings.Default.InputsFilePath));
            }

            if (String.IsNullOrEmpty(Settings.Default.Function))
            {
                throw new ApplicationException("You must define Activate Function in configuration file!.");
            }
        }
    }
}
