using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using NeuronNetwork.Base;

namespace NeuronNetwork.Utility
{
    public static class DataLoader
    {
        public const string CountCommand = "#NeuronsCount";

        public const string EndCommand = "#End";

        public const string EndConfigurationCommand = "#EndConfig";
        public const string EndLayerCommand = "#EndLayer";
        public const string NewLayerCommand = "#NewLayer";
        public const string StartConfigurationCommand = "#StartConfig";
        public const string WeightsCommand = "#Weights";
		public const string ThresholdCommand = "#Threshold";

        public static void Load(string fileName, out List<LayerData> layers)
        {
            var fileInfo = new FileInfo(fileName);

            layers = new List<LayerData>();

            using (StreamReader reader = fileInfo.OpenText())
            {
                LoadConfiguration(layers, reader);
            }
        }

        private static void LoadConfiguration(ICollection<LayerData> layers, TextReader reader)
        {
            string text;
            do
            {
                text = reader.ReadLine();
                switch (text)
                {
                    case StartConfigurationCommand:
                        break;
                    case NewLayerCommand:
                        layers.Add(CreateNewLayerFromFile(reader));
                        break;
                }
            } while (text != EndConfigurationCommand);

            reader.Close();
        }

        private static LayerData CreateNewLayerFromFile(TextReader reader)
        {
            string text;
            LayerData layerData = new LayerData();
			int index = 0;

            while (!String.Equals(text = reader.ReadLine(), EndLayerCommand))
            {
                switch (text)
                {
                    case WeightsCommand:
						layerData.Weights.Add(new List<double>());
                        LoadWeights(layerData.Weights[index], reader);
						index++;
                        break;
					case ThresholdCommand:
						layerData.Thresholds.Add(LoadThreshold(reader));
						
						break;
                    case CountCommand:
                        layerData.CountNeurons = LoadCount(reader);
                        break;
                }
            }

            return layerData;
        }

		private static double LoadThreshold(TextReader reader){
            NumberFormatInfo nfi = new CultureInfo("en-US", false).NumberFormat;
			double result = Double.Parse(reader.ReadLine(), nfi);

            if (String.Equals(reader.ReadLine(), EndCommand))
            {
                return result;
            }

            throw new InvalidDataException("After threshold value in the file you must type end command - #End.");
		}
		
        private static int LoadCount(TextReader reader)
        {
            int result = Int32.Parse(reader.ReadLine());

            if (String.Equals(reader.ReadLine(), EndCommand))
            {
                return result;
            }

            throw new InvalidDataException("After count number in the file you must type end command - #End.");
        }

        private static void LoadWeights(ICollection<double> weights, TextReader streamReader)
        {
            string text;

            while (!String.Equals(text = streamReader.ReadLine(), EndCommand))
            {
                NumberFormatInfo nfi = new CultureInfo("en-US", false).NumberFormat;
                weights.Add(Convert.ToDouble(text, nfi));
            }
        }

        #region Nested type: LayerData

        #endregion

        public static List<double> ReadInputs(string fileName)
        {
            List<double> results = new List<double>();
            var fileInfo = new FileInfo(fileName);

            using (StreamReader reader = fileInfo.OpenText())
            {
                string text;
                while (!String.Equals(text = reader.ReadLine(), null))
                {
                    NumberFormatInfo nfi = new CultureInfo("en-US", false).NumberFormat;
                    results.Add(Convert.ToDouble(text, nfi));
                }
            }

            return results;
        }
    }
}