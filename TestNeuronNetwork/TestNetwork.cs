using NeuronNetwork.Base.Networks;
using NeuronNetwork.Utility;
using NeuronNetwork.Base.Neurons;
using NeuronNetwork.Base.Layers;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using NeuronNetwork.Base.Learning;

namespace TestNeuronNetwork
{
    [TestFixture]
    public class TestNetwork
    {
        private INetwork _network;
		private INetwork _network_AND_00;
		private INetwork _network_AND_01;
		private INetwork _network_AND_10;
		private INetwork _network_AND_11;
		private INetwork _network_Threshold_AND_11;
		private INetwork _network_Threshold_AND_10;
		private INetwork _network_Threshold_AND_01;
		private INetwork _network_Threshold_AND_00;
        private INetwork _network_Threshold_AND_08;
		private INetwork _networkXOR11;
		private INetwork _networkXOR01;
		private INetwork _networkXOR10;
		private INetwork _networkXOR00;
		private INetwork _networkKohonen;
		private INetwork _networkKohonen2;
		private INetwork _networkKohonen3;
		private INetwork _networkKohonen_parz;
		private INetwork _network_XOR_BP;
		private INetwork _network_ikonki_BP;
		
		private INetwork _network_parz_BP;
		
		private KohonenLearning _kohonenLearning;
		private KohonenLearning _kohonenLearning_parz;
		private BackPropagationLearning _backPropagationLearning_parz;
		private BackPropagationLearning _backPropagationLearning_XOR;
		private BackPropagationLearning _backPropagationLearning_ikonki;
		
        private NeuralNetworkFileParser _parser;
        private const int InputsCount = 4;
		private const int Inputs_AND_Count = 3;

        [SetUp]
        public void SetUp()
        {
            this._parser = new NeuralNetworkFileParser();
			
            this._network = this._parser.CreateNetwork("Configuration_AND.txt", "sigmoid", InputsCount);
			this._network_AND_00 = this._parser.CreateNetwork("Configuration_AND.txt", "linear", Inputs_AND_Count);
			this._network_AND_01 = this._parser.CreateNetwork("Configuration_AND.txt", "linear", Inputs_AND_Count);
			this._network_AND_10 = this._parser.CreateNetwork("Configuration_AND.txt", "linear", Inputs_AND_Count);
			this._network_AND_11 = this._parser.CreateNetwork("Configuration_AND.txt", "linear", Inputs_AND_Count);
			this._network_Threshold_AND_11 = this._parser.CreateNetwork("Configuration_AND_threshold.txt","threshold",Inputs_AND_Count - 1);
			this._network_Threshold_AND_10 = this._parser.CreateNetwork("Configuration_AND_threshold.txt","threshold",Inputs_AND_Count - 1);
            this._network_Threshold_AND_08 = this._parser.CreateNetwork("Configuration_AND_threshold.txt", "threshold", Inputs_AND_Count - 1);
			this._network_Threshold_AND_01 = this._parser.CreateNetwork("Configuration_AND_threshold.txt","threshold",Inputs_AND_Count - 1);
			this._network_Threshold_AND_00 = this._parser.CreateNetwork("Configuration_AND_threshold.txt","threshold",Inputs_AND_Count - 1);
        	this._networkXOR00 = this._parser.CreateNetwork("Configuration_XOR_threshold.txt","threshold",2);
			this._networkXOR01 = this._parser.CreateNetwork("Configuration_XOR_threshold.txt","threshold",2);
			this._networkXOR10 = this._parser.CreateNetwork("Configuration_XOR_threshold.txt","threshold",2);
			this._networkXOR11 = this._parser.CreateNetwork("Configuration_XOR_threshold.txt","threshold",2);
			
			
			this._networkKohonen = this._parser.CreateNetwork("Configuration_Kohonen.txt", "linear",4);
			this._networkKohonen2 = this._parser.CreateNetwork("Configuration_Kohonen2.txt", "linear",9);
			this._networkKohonen3 = this._parser.CreateNetwork("Configuration_Kohonen3.txt", "linear",9);
			this._networkKohonen_parz = this._parser.CreateNetwork("Configuration_Kohonen_parz.txt", "linear",3);
			this._network_parz_BP = this._parser.CreateNetwork("Configuration_Kohonen_parz_BP.txt", "sigmoid",3);
			this._network_XOR_BP = this._parser.CreateNetwork("Configuration_Kohonen_XOR_BP.txt", "sigmoid",2);
			this._network_ikonki_BP = this._parser.CreateNetwork("Configuration_ikonki_BP.txt", "sigmoid",9);
			
		}

//        [Test]
        public void TestAfterInit()
        {
            Assert.AreEqual(InputsCount, this._network.InputsCount);

            Assert.AreEqual(3, this._network.Layers.Count);

            Assert.AreEqual(4, this._network.Layers[0].Neurons.Count);
            Assert.AreEqual(5, this._network.Layers[1].Neurons.Count);
            Assert.AreEqual(6, this._network.Layers[2].Neurons.Count);
        }

		
		public void ShowOut(IList<double> input){
	
			int i = 1;
			
			foreach(double d in input){
				Console.WriteLine(i + " " +  d);
				i++;
			}
		}
		
		
		
		public void TestKohonen1()
        {
			//List<List<double>> input = new List<List<double>>();
			
			/*
			//wektor 1
			List<double> w1 = new List<double> {1, 1, 0, 0};

		    //wektor 2
			List<double> w2 = new List<double> {0, 0, 0, 1};

		    //wektor 3
			List<double> w3 = new List<double> {1, 0, 0, 0};

		    //wekror 4
			List<double> w4 = new List<double> {0, 0, 1, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			*/
			
			double[] w1 = {1, 1, 0, 0};
			double[] w2 = {0, 0, 0, 1};
			double[] w3 = {1, 0, 0, 0};
			double[] w4 = {0, 0, 1, 1};
			
			double[][] input = new double[][]{w1,w2,w3,w4};
			
			/*
			this._networkKohonen.Layers[0].Alpha = 0.6;
            this._networkKohonen.Layers[0].Learn(100,input);
			*/
			
			_kohonenLearning = new KohonenLearning(this._networkKohonen.Layers[0]);
			_kohonenLearning.LearningRate = 0.13;
			_kohonenLearning.LearningRadius = 0;
			
			for(int i=0;i<4000;i++){
				
				_kohonenLearning.RunEpoch(input);
			
			}
			this._networkKohonen.Compute(DataLoader.ReadInputs("Inputs_Kohonen.txt"));
		
			
			
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen.Layers[0].Neurons;
			
			foreach (INeuron n in neurony){
				foreach (double val in n.Weights)
					Console.WriteLine(val);
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen.Outputs[0] + " " + this._networkKohonen.Outputs[1]);
				Assert.Greater(this._networkKohonen.Outputs[1],this._networkKohonen.Outputs[0] );
           
			
        }
		
		[Test]
		public void TestBP_XOR()
        {
			
			double[] w1 = {0.0, 0.0};
			double[] w2 = {0, 1.0};
			double[] w3 = {1.0, 0.0};
			
			double[] w4 = { 1.0, 1.0};
			
			
			double[][] input = new double[][]{w1,w2,w3,w4};
			
			_backPropagationLearning_XOR = new BackPropagationLearning( _network_XOR_BP );
			
			double[] out1 = {0.0};
			double[] out2 = {1.0};
			double[] out3 = {1.0};
			double[] out4 = {0.0};
				
			double[][] output = new double[][]{out1, out2, out3,out4};
			
			_backPropagationLearning_XOR.LearningRate = 0.5;
			
			for (int i = 0 ; i < 20000; i++){
				
				
				_backPropagationLearning_XOR.LearningRate *= 0.999;
					
				int index = 0;
				
				foreach(double[] wektor in  input){
				
					this._network_XOR_BP.Layers[0].Compute(new List<double>(wektor));
					IList<double> BPInput = this._network_XOR_BP.Layers[0].Outputs; 
					
					//double[] BPInputArray = new double[40];
					
					//BPInput.CopyTo(BPInputArray,0);
					
					Console.WriteLine( _backPropagationLearning_XOR.Run(wektor, output[index]) );
				
					index++;
				}
				
			}
			
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._network_XOR_BP.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine("----------------------WARSTWA DRUGA WAGI -----------------------------");
			
			 neurony = this._network_XOR_BP.Layers[1].Neurons;
			
		 counter = 0;
			foreach (INeuron n in neurony)
            {
				
				Console.WriteLine("THRESHOLD: " +  n.Threshold);
				
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			
			this._network_XOR_BP.Compute(DataLoader.ReadInputs("00.txt"));
			
			Console.WriteLine("00:")	;
			ShowOut(this._network_XOR_BP.Outputs);
			
			
			this._network_XOR_BP.Compute(DataLoader.ReadInputs("01.txt"));
			
			Console.WriteLine("01:")	;
			ShowOut(this._network_XOR_BP.Outputs);
			
			this._network_XOR_BP.Compute(DataLoader.ReadInputs("10.txt"));
			
			Console.WriteLine("10:")	;
			ShowOut(this._network_XOR_BP.Outputs);
			
			this._network_XOR_BP.Compute(DataLoader.ReadInputs("11.txt"));
			
			Console.WriteLine("11:")	;
			ShowOut(this._network_XOR_BP.Outputs);
			
			
			}
	
		[Test]
		public void TestBP_ikonki()
        {
			
			
			double[] w3 = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, 0, 0.0};
			double[] w2 = {1.0, 0, 0, 0.0, 1.0, 0.0, 0, 0.0, 1.0};
			double[] w1 = {0.0, 0.0, 1.0, 0, 1.0, 0.0, 1.0, 0.0, 0.0};
			
			
			
			double[][] input = new double[][]{w1,w2,w3};
			
	
			_backPropagationLearning_ikonki = new BackPropagationLearning(_network_ikonki_BP);
				_backPropagationLearning_ikonki.LearningRate = 0.2;
			
			double[] out1 = {1.0, 0.0, 0.0};
			double[] out2 = {0.0, 1.0, 0.0};
			double[] out3 = {0.0, 0.0, 1.0};
				
			double[][] output = new double[][]{out1, out2, out3};
			
				
		for (int i = 0 ; i < 20000; i++){
				
				
				_backPropagationLearning_ikonki.LearningRate *= 0.999;
					
				int index = 0;
				
				foreach(double[] wektor in  input){
				
					this._network_ikonki_BP.Layers[0].Compute(new List<double>(wektor));
					IList<double> BPInput = this._network_XOR_BP.Layers[0].Outputs; 
					
					//double[] BPInputArray = new double[40];
					
					//BPInput.CopyTo(BPInputArray,0);
					
					Console.WriteLine( _backPropagationLearning_ikonki.Run(wektor, output[index]) );
				
					index++;
				}
				
			}
			
			
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._network_ikonki_BP.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine("----------------------WARSTWA DRUGA WAGI -----------------------------");
			
			 neurony = this._network_ikonki_BP.Layers[1].Neurons;
			
		 counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("ikonki1.txt"));
			
			Console.WriteLine("ikonki1:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("ikonki2.txt"));
			
			Console.WriteLine("ikonki2:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("ikonki3.txt"));
			
			Console.WriteLine("ikonki3:");
			ShowOut(this._networkKohonen3.Outputs);
			
			}
		
	
	
		
		
	//	[Test]
		public void TestBP_parz()
        {
			
			double[] w1 = {0.0, 0.0, 0.0};
			double[] w2 = {0, 0, 1.0};
			double[] w3 = {0, 1.0, 0};
			
			double[] w4 = {0.0, 1.0, 1.0};
			double[] w5 = {1.0, 0.0, 0};
			double[] w6 = {1.0, 0, 1.0};
			
			double[] w7 = {1.0, 1.0, 0};
			double[] w8 = {1.0, 1.0, 1.0};
			
			
			double[][] input = new double[][]{w1,w2,w3,w4,w5,w6,w7,w8};
			
			
			/*
			this._networkKohonen2.Layers[0].Alpha = 0.12;
			this._networkKohonen2.Layers[0].Dim = 0;
            this._networkKohonen2.Layers[0].Learn(5000,input);
						
			
			_kohonenLearning_parz = new KohonenLearning(this._networkKohonen_parz.Layers[0]);
			_kohonenLearning_parz.LearningRate = 0.12;
			_kohonenLearning_parz.LearningRadius = 0;
			
			for(int i=0;i<5000;i++){
				_kohonenLearning_parz.LearningRate *= 0.999;
				_kohonenLearning_parz.RunEpoch(input);
			
			}
			
			WidrowHoffLearning _widrowHoff = new WidrowHoffLearning(this._networkKohonen_parz.Layers[1]);
			*/
			
			_backPropagationLearning_parz = new BackPropagationLearning(_network_parz_BP);
			
			double[] out1 = {0.0,1.0};
			double[] out2 = {1.0, 0.0};
			double[] out3 = {1.0, 0.0};
			double[] out4 = {0.0,1.0};
			double[] out5 = {1.0, 0.0};
			double[] out6 = {0.0,1.0};
			double[] out7 = {0.0,1.0};
			double[] out8 = {1.0, 0.0};
				
			double[][] output = new double[][]{out1, out2, out3,out4,out5, out6, out7, out8};
			
			/*
			_widrowHoff.LearningRate = 0.6;
			
			for(int i=0;i<200;i++){
				_widrowHoff.LearningRate *= 0.99;
				
				int index = 0;
				
				foreach(double[] wektor in  input){
						
					this._networkKohonen_parz.Layers[0].Compute(new List<double>(wektor));
					IList<double> widrowInput = this._networkKohonen_parz.Layers[0].Outputs; 
					
					double[] widrowInputArray = new double[8];
					
					widrowInput.CopyTo(widrowInputArray,0);
					
				//	foreach(double d in widrowInputArray)
				//		Console.WriteLine(d + " " );
						
					
					_widrowHoff.Run(widrowInputArray, output[index]);
					//Console.WriteLine( _widrowHoff.Run(widrowInputArray, output[index%3]));
					index++;
				}
			
			}
			*/
			
			_backPropagationLearning_parz.LearningRate = 0.2;
			
			for (int i = 0 ; i < 20000; i++){
				
				
				_backPropagationLearning_parz.LearningRate *= 0.999;
					
				int index = 0;
				
				foreach(double[] wektor in  input){
				
					this._network_parz_BP.Layers[0].Compute(new List<double>(wektor));
					IList<double> BPInput = this._network_parz_BP.Layers[0].Outputs; 
					
					//double[] BPInputArray = new double[40];
					
					//BPInput.CopyTo(BPInputArray,0);
					
					Console.WriteLine( _backPropagationLearning_parz.Run(wektor, output[index]) );
				
					index++;
				}
				
			}
			
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._network_parz_BP.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 10 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine("----------------------WARSTWA DRUGA WAGI -----------------------------");
			
			 neurony = this._network_parz_BP.Layers[1].Neurons;
			
		 counter = 0;
			foreach (INeuron n in neurony)
            {
				
				Console.WriteLine("THRESHOLD: " +  n.Threshold);
				
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 10 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			
			this._network_parz_BP.Compute(DataLoader.ReadInputs("000.txt"));
			
			Console.WriteLine("000:")	;
			ShowOut(this._network_parz_BP.Outputs);
			
			this._network_parz_BP.Compute(DataLoader.ReadInputs("001.txt"));
			
			Console.WriteLine("001:")	;
			ShowOut(this._network_parz_BP.Outputs);
			
			this._network_parz_BP.Compute(DataLoader.ReadInputs("010.txt"));
			
			Console.WriteLine("010:")	;
			ShowOut(this._network_parz_BP.Outputs);
			
			this._network_parz_BP.Compute(DataLoader.ReadInputs("011.txt"));
			
			Console.WriteLine("011:")	;
			ShowOut(this._network_parz_BP.Outputs);
			
			this._network_parz_BP.Compute(DataLoader.ReadInputs("100.txt"));
			
			Console.WriteLine("100:")	;
			ShowOut(this._network_parz_BP.Outputs);
			
			this._network_parz_BP.Compute(DataLoader.ReadInputs("101.txt"));
			
			Console.WriteLine("101:")	;
			ShowOut(this._network_parz_BP.Outputs);
			
			this._network_parz_BP.Compute(DataLoader.ReadInputs("110.txt"));
			
			Console.WriteLine("110:")	;
			ShowOut(this._network_parz_BP.Outputs);
			
			this._network_parz_BP.Compute(DataLoader.ReadInputs("111.txt"));
			
			Console.WriteLine("111:")	;
			ShowOut(this._network_parz_BP.Outputs);
			
			}
		
		
		
		
		
		
		
		//[Test]
		public void TestCP_parz()
        {
			
			
			
			double[] w1 = {0.0, 0.0, 0.0};
			double[] w2 = {0, 0, 1.0};
			double[] w3 = {0, 1.0, 0};
			
			double[] w4 = {0.0, 1.0, 1.0};
			double[] w5 = {1.0, 0.0, 0};
			double[] w6 = {1.0, 0, 1.0};
			
			double[] w7 = {1.0, 1.0, 0};
			double[] w8 = {1.0, 1.0, 1.0};
			
			
			
			double[][] input = new double[][]{w1,w2,w3,w4,w5,w6,w7,w8};
			
			
			/*
			this._networkKohonen2.Layers[0].Alpha = 0.12;
			this._networkKohonen2.Layers[0].Dim = 0;
            this._networkKohonen2.Layers[0].Learn(5000,input);
			*/			
			
			_kohonenLearning_parz = new KohonenLearning(this._networkKohonen_parz.Layers[0]);
			_kohonenLearning_parz.LearningRate = 0.12;
			_kohonenLearning_parz.LearningRadius = 0;
			
			for(int i=0;i<5000;i++){
				_kohonenLearning_parz.LearningRate *= 0.999;
				_kohonenLearning_parz.RunEpoch(input);
			
			}
			
			WidrowHoffLearning _widrowHoff = new WidrowHoffLearning(this._networkKohonen_parz.Layers[1]);
			
			double[] out1 = {0.0,1.0};
			double[] out2 = {1.0, 0.0};
			double[] out3 = {1.0, 0.0};
			double[] out4 = {0.0,1.0};
			double[] out5 = {1.0, 0.0};
			double[] out6 = {0.0,1.0};
			double[] out7 = {0.0,1.0};
			double[] out8 = {1.0, 0.0};
				
			double[][] output = new double[][]{out1, out2, out3,out4,out5, out6, out7, out8};
			
			_widrowHoff.LearningRate = 0.6;
			
			for(int i=0;i<200;i++){
				_widrowHoff.LearningRate *= 0.99;
				
				int index = 0;
				
				foreach(double[] wektor in  input){
						
					this._networkKohonen_parz.Layers[0].Compute(new List<double>(wektor));
					IList<double> widrowInput = this._networkKohonen_parz.Layers[0].Outputs; 
					
					double[] widrowInputArray = new double[8];
					
					widrowInput.CopyTo(widrowInputArray,0);
					
				//	foreach(double d in widrowInputArray)
				//		Console.WriteLine(d + " " );
						
					
					_widrowHoff.Run(widrowInputArray, output[index]);
					//Console.WriteLine( _widrowHoff.Run(widrowInputArray, output[index%3]));
					index++;
				}
			
			}
			
			
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen_parz.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine("----------------------WARSTWA DRUGA WAGI -----------------------------");
			
			 neurony = this._networkKohonen_parz.Layers[1].Neurons;
			
		 counter = 0;
			foreach (INeuron n in neurony)
            {
				
				Console.WriteLine("THRESHOLD: " +  n.Threshold);
				
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			
			this._networkKohonen_parz.Compute(DataLoader.ReadInputs("000.txt"));
			
			Console.WriteLine("000:")	;
			ShowOut(this._networkKohonen_parz.Outputs);
			
			this._networkKohonen_parz.Compute(DataLoader.ReadInputs("001.txt"));
			
			Console.WriteLine("001:")	;
			ShowOut(this._networkKohonen_parz.Outputs);
			
			this._networkKohonen_parz.Compute(DataLoader.ReadInputs("010.txt"));
			
			Console.WriteLine("010:")	;
			ShowOut(this._networkKohonen_parz.Outputs);
			
			this._networkKohonen_parz.Compute(DataLoader.ReadInputs("011.txt"));
			
			Console.WriteLine("011:")	;
			ShowOut(this._networkKohonen_parz.Outputs);
			
			this._networkKohonen_parz.Compute(DataLoader.ReadInputs("100.txt"));
			
			Console.WriteLine("100:")	;
			ShowOut(this._networkKohonen_parz.Outputs);
			
			this._networkKohonen_parz.Compute(DataLoader.ReadInputs("101.txt"));
			
			Console.WriteLine("101:")	;
			ShowOut(this._networkKohonen_parz.Outputs);
			
			this._networkKohonen_parz.Compute(DataLoader.ReadInputs("110.txt"));
			
			Console.WriteLine("110:")	;
			ShowOut(this._networkKohonen_parz.Outputs);
			
			this._networkKohonen_parz.Compute(DataLoader.ReadInputs("111.txt"));
			
			Console.WriteLine("111:")	;
			ShowOut(this._networkKohonen_parz.Outputs);
			
			}
		
		
		
		//[Test]
		public void TestCP_2()
        {
			
			
			double[] w1 = {1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 1.0};
			double[] w2 = {1.0, 0, 0, 1.0, 0.0, 0.0, 0, 1.0, 1.0};
			double[] w3 = {1.0, 1.0, 0, 0, 0, 1.0, 0.0, 0.0, 1.0};
			
			double[] w4 = {1.0, 0, 1.0, 0.0, 1.0, 0, 1.0, 0, 1.0};
			double[] w5 = {0, 1.0, 1.0, 0, 1.0, 0, 1.0, 1.0, 0};
			double[] w6 = {1.0, 0, 0.0, 1.0, 1.0, 1.0, 0, 0, 1.0};
			
			double[] w7 = {1, 1.0, 1.0, 1.0, 0, 0, 1.0, 0, 0};
			double[] w8 = {0, 1.0, 0.0, 0, 1.0, 0, 1.0, 1.0, 1.0};
			double[] w9 = {1.0, 1.0, 1.0, 0, 0, 1.0, 0, 0, 1};
			
			
			double[][] input = new double[][]{w1,w2,w3,w4,w5,w6,w7,w8,w9};
			
			
			/*
			this._networkKohonen2.Layers[0].Alpha = 0.12;
			this._networkKohonen2.Layers[0].Dim = 0;
            this._networkKohonen2.Layers[0].Learn(5000,input);
			*/			
			
			_kohonenLearning = new KohonenLearning(this._networkKohonen3.Layers[0]);
			_kohonenLearning.LearningRate = 0.02;
			_kohonenLearning.LearningRadius = 0;
			
			for(int i=0;i<5000;i++){
				_kohonenLearning.LearningRate *= 0.999;
				_kohonenLearning.RunEpoch(input);
			
			}
			
			WidrowHoffLearning _widrowHoff = new WidrowHoffLearning(this._networkKohonen3.Layers[1]);
			
			double[] out1 = {1.0, 0.0, 0.0};
			double[] out2 = {0.0, 1.0, 0.0};
			double[] out3 = {0.0, 0.0, 1.0};
				
			double[][] output = new double[][]{out1, out2, out3};
			
			_widrowHoff.LearningRate = 0.06;
			
			for(int i=0;i<30;i++){
				_widrowHoff.LearningRate *= 0.99;
				
				int index = 0;
				
				foreach(double[] wektor in  input){
						
					this._networkKohonen3.Layers[0].Compute(new List<double>(wektor));
					IList<double> widrowInput = this._networkKohonen3.Layers[0].Outputs; 
					
					double[] widrowInputArray = new double[9];
					
					widrowInput.CopyTo(widrowInputArray,0);
					
				//	foreach(double d in widrowInputArray)
				//		Console.WriteLine(d + " " );
						
					
					_widrowHoff.Run(widrowInputArray, output[index%3]);
					//Console.WriteLine( _widrowHoff.Run(widrowInputArray, output[index%3]));
					index++;
				}
			
			}
			
			
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen3.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine("----------------------WARSTWA DRUGA WAGI -----------------------------");
			
			 neurony = this._networkKohonen3.Layers[1].Neurons;
			
		 counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("test_2_1.txt"));
			
			Console.WriteLine("test 2_1:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("test_2_2.txt"));
			
			Console.WriteLine("test 2_2:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("test_2_3.txt"));
			
			Console.WriteLine("test 2_3:");
			ShowOut(this._networkKohonen3.Outputs);
			
			}
		
		
		
		
		
		//[Test]
		public void TestCP()
        {
			
			/*
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {1, 1, 1, 0, 0, 0, 0, 0, 0};


		    //wektor 2
			List<double> w2 = new List<double> {0, 0, 0, 1, 1, 1, 0, 0, 0};

		    //wektor 3
			List<double> w3 = new List<double>  {0, 0, 0, 0, 0, 0, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 1, 0, 0, 1, 0, 0};

			//wekror 5
			List<double> w5 = new List<double> {0, 1, 0, 0, 1, 0, 0, 1, 0};
			//wekror 6
			List<double> w6 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};
			//wekror 7
			List<double> w7 = new List<double> {1, 0, 0, 0, 0, 0, 0, 0, 0};
			//wekror 8
			List<double> w8 = new List<double> {0, 0, 1, 0, 1, 0, 1, 0, 0};
			//wekror 9
			List<double> w9 = new List<double> {0, 0, 0, 0, 0, 0, 0, 0, 1};
		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			input.Add(w5);
			input.Add(w6);
			input.Add(w7);
			input.Add(w8);
			input.Add(w9);
			*/
			
			double[] w1 = {1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0};
			double[] w2 = {0, 0, 0, 1.0, 1.0, 1.0, 0, 0, 0};
			double[] w3 = {0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0};
			
			double[] w4 = {1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0};
			double[] w5 = {0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0};
			double[] w6 = {0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0};
			
			double[] w7 = {1, 0, 0, 0, 0, 0, 0, 0, 0};
			double[] w8 = {0, 0, 1.0, 0, 1.0, 0, 1.0, 0, 0};
			double[] w9 = {0, 0, 0, 0, 0, 0, 0, 0, 1};
			
			
			double[][] input = new double[][]{w1,w2,w3,w4,w5,w6,w7,w8,w9};
			
			
			/*
			this._networkKohonen2.Layers[0].Alpha = 0.12;
			this._networkKohonen2.Layers[0].Dim = 0;
            this._networkKohonen2.Layers[0].Learn(5000,input);
			*/			
			
			_kohonenLearning = new KohonenLearning(this._networkKohonen3.Layers[0]);
			_kohonenLearning.LearningRate = 0.02;
			_kohonenLearning.LearningRadius = 0;
			
			for(int i=0;i<5000;i++){
				_kohonenLearning.LearningRate *= 0.999;
				_kohonenLearning.RunEpoch(input);
			
			}
			
			WidrowHoffLearning _widrowHoff = new WidrowHoffLearning(this._networkKohonen3.Layers[1]);
			
			double[] out1 = {1.0, 0.0, 0.0};
			double[] out2 = {0.0, 1.0, 0.0};
			double[] out3 = {0.0, 0.0, 1.0};
				
			double[][] output = new double[][]{out1, out2, out3};
			
			_widrowHoff.LearningRate = 0.06;
			
			for(int i=0;i<30;i++){
				_widrowHoff.LearningRate *= 0.99;
				
				int index = 0;
				
				foreach(double[] wektor in  input){
						
					this._networkKohonen3.Layers[0].Compute(new List<double>(wektor));
					IList<double> widrowInput = this._networkKohonen3.Layers[0].Outputs; 
					
					double[] widrowInputArray = new double[9];
					
					widrowInput.CopyTo(widrowInputArray,0);
					
				//	foreach(double d in widrowInputArray)
				//		Console.WriteLine(d + " " );
						
					
					_widrowHoff.Run(widrowInputArray, output[index%3]);
					//Console.WriteLine( _widrowHoff.Run(widrowInputArray, output[index%3]));
					index++;
				}
			
			}
			
			
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen3.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine("----------------------WARSTWA DRUGA WAGI -----------------------------");
			
			 neurony = this._networkKohonen3.Layers[1].Neurons;
			
		 counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_1_1.txt"));
			
			Console.WriteLine("klasa 1_1:");
			//Console.WriteLine(this._networkKohonen3.Outputs[0] + "   " + this._networkKohonen3.Outputs[1]);
			//Console.WriteLine(this._networkKohonen3.Outputs[2]  ); //+  " " + this._networkKohonen3.Outputs[3]);
			
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_1_2.txt"));
			
			Console.WriteLine("klasa 1_2:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_1_3.txt"));
			
			Console.WriteLine("klasa 1_3:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_2_1.txt"));
			
			Console.WriteLine("klasa 2_1:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_2_2.txt"));
			
			Console.WriteLine("klasa 2_2:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_2_3.txt"));
			
			Console.WriteLine("klasa 2_3:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_3_1.txt"));
			
			Console.WriteLine("klasa 3_1:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_3_2.txt"));
			
			Console.WriteLine("klasa 3_2:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("klasa_3_3.txt"));
			
			Console.WriteLine("klasa 3_3:");
			ShowOut(this._networkKohonen3.Outputs);
			
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("test_1.txt"));
			
			Console.WriteLine("test 1:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("test_2.txt"));
			
			Console.WriteLine("test 2:");
			ShowOut(this._networkKohonen3.Outputs);
			
			this._networkKohonen3.Compute(DataLoader.ReadInputs("test_3.txt"));
			
			Console.WriteLine("test 3:");
			ShowOut(this._networkKohonen3.Outputs);
			
		//	Console.WriteLine(this._networkKohonen3.Layers[1].);
			
			}
		
	//	[Test]
		public void TestKohonen_1D_1000E()
        {
			
			/*
			 * 
			 * Alpha = 0.13
			 * sasiedztwo 1D
			 * 1000 epok uczenia
			 * 
			 */
			
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};


		    //wektor 2
			List<double> w2 = new List<double> {0, 1, 0, 1, 1, 1, 0, 1, 0};

		    //wektor 3
			List<double> w3 = new List<double> {1, 1, 1, 1, 0, 1, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 0, 1, 0, 0, 0, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			
			this._networkKohonen2.Layers[0].Alpha = 0.13;
			this._networkKohonen2.Layers[0].Dim = 1;
            this._networkKohonen2.Layers[0].Learn(1000,input);

			this._networkKohonen2.Compute(DataLoader.ReadInputs("Inputs_Kohonen2.txt"));
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen2.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen2.Outputs[0] + "   " + this._networkKohonen2.Outputs[1]);
			Console.WriteLine(this._networkKohonen2.Outputs[2] + "   " + this._networkKohonen2.Outputs[3]);
        }
		
	//	[Test]
		public void TestKohonen_1D_4000E()
        {
			
			/*
			 * 
			 * Alpha = 0.13
			 * sasiedztwo 1D
			 * 4000 epok uczenia
			 * 
			 */
			
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};


		    //wektor 2
			List<double> w2 = new List<double> {0, 1, 0, 1, 1, 1, 0, 1, 0};

		    //wektor 3
			List<double> w3 = new List<double> {1, 1, 1, 1, 0, 1, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 0, 1, 0, 0, 0, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			
			this._networkKohonen2.Layers[0].Alpha = 0.13;
			this._networkKohonen2.Layers[0].Dim = 1;
            this._networkKohonen2.Layers[0].Learn(4000,input);

			this._networkKohonen2.Compute(DataLoader.ReadInputs("Inputs_Kohonen2.txt"));
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen2.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen2.Outputs[0] + "   " + this._networkKohonen2.Outputs[1]);
			Console.WriteLine(this._networkKohonen2.Outputs[2] + "   " + this._networkKohonen2.Outputs[3]);
        }
		
//		[Test]
		public void TestKohonen_1D_8000E()
        {
			
			/*
			 * 
			 * Alpha = 0.13
			 * sasiedztwo 1D
			 * 8000 epok uczenia
			 * 
			 */
			
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};


		    //wektor 2
			List<double> w2 = new List<double> {0, 1, 0, 1, 1, 1, 0, 1, 0};

		    //wektor 3
			List<double> w3 = new List<double> {1, 1, 1, 1, 0, 1, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 0, 1, 0, 0, 0, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			
			this._networkKohonen2.Layers[0].Alpha = 0.13;
			this._networkKohonen2.Layers[0].Dim = 1;
            this._networkKohonen2.Layers[0].Learn(8000,input);

			this._networkKohonen2.Compute(DataLoader.ReadInputs("Inputs_Kohonen2.txt"));
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen2.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen2.Outputs[0] + "   " + this._networkKohonen2.Outputs[1]);
			Console.WriteLine(this._networkKohonen2.Outputs[2] + "   " + this._networkKohonen2.Outputs[3]);
        }
		
	//	[Test]
		public void TestKohonen_2D_1000E()
        {
			
			/*
			 * 
			 * Alpha = 0.13
			 * sasiedztwo 2D
			 * 1000 epok uczenia
			 * 
			 */
			
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};


		    //wektor 2
			List<double> w2 = new List<double> {0, 1, 0, 1, 1, 1, 0, 1, 0};

		    //wektor 3
			List<double> w3 = new List<double> {1, 1, 1, 1, 0, 1, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 0, 1, 0, 0, 0, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			
			this._networkKohonen2.Layers[0].Alpha = 0.13;
			this._networkKohonen2.Layers[0].Dim = 2;
            this._networkKohonen2.Layers[0].Learn(1000,input);

			this._networkKohonen2.Compute(DataLoader.ReadInputs("Inputs_Kohonen2.txt"));
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen2.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen2.Outputs[0] + "   " + this._networkKohonen2.Outputs[1]);
			Console.WriteLine(this._networkKohonen2.Outputs[2] + "   " + this._networkKohonen2.Outputs[3]);
        }
		
	//	[Test]
		public void TestKohonen_2D_4000E()
        {
			
			/*
			 * 
			 * Alpha = 0.13
			 * sasiedztwo 2D
			 * 4000 epok uczenia
			 * 
			 */
			
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};


		    //wektor 2
			List<double> w2 = new List<double> {0, 1, 0, 1, 1, 1, 0, 1, 0};

		    //wektor 3
			List<double> w3 = new List<double> {1, 1, 1, 1, 0, 1, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 0, 1, 0, 0, 0, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			
			this._networkKohonen2.Layers[0].Alpha = 0.13;
			this._networkKohonen2.Layers[0].Dim = 2;
            this._networkKohonen2.Layers[0].Learn(4000,input);

			this._networkKohonen2.Compute(DataLoader.ReadInputs("Inputs_Kohonen2.txt"));
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen2.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen2.Outputs[0] + "   " + this._networkKohonen2.Outputs[1]);
			Console.WriteLine(this._networkKohonen2.Outputs[2] + "   " + this._networkKohonen2.Outputs[3]);
        }
		
	//	[Test]
		public void TestKohonen_2D_8000E()
        {
			
			/*
			 * 
			 * Alpha = 0.13
			 * sasiedztwo 2D
			 * 8000 epok uczenia
			 * 
			 */
			
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};


		    //wektor 2
			List<double> w2 = new List<double> {0, 1, 0, 1, 1, 1, 0, 1, 0};

		    //wektor 3
			List<double> w3 = new List<double> {1, 1, 1, 1, 0, 1, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 0, 1, 0, 0, 0, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			
			this._networkKohonen2.Layers[0].Alpha = 0.13;
			this._networkKohonen2.Layers[0].Dim = 2;
            this._networkKohonen2.Layers[0].Learn(8000,input);

			this._networkKohonen2.Compute(DataLoader.ReadInputs("Inputs_Kohonen2.txt"));
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen2.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen2.Outputs[0] + "   " + this._networkKohonen2.Outputs[1]);
			Console.WriteLine(this._networkKohonen2.Outputs[2] + "   " + this._networkKohonen2.Outputs[3]);
        }
		
	//	[Test]
		public void TestKohonen_0D_1000E()
        {
			
			/*
			 * 
			 * Alpha = 0.13
			 * sasiedztwo brak
			 * 1000 epok uczenia
			 * 
			 */
			
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};


		    //wektor 2
			List<double> w2 = new List<double> {0, 1, 0, 1, 1, 1, 0, 1, 0};

		    //wektor 3
			List<double> w3 = new List<double> {1, 1, 1, 1, 0, 1, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 0, 1, 0, 0, 0, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			
			this._networkKohonen2.Layers[0].Alpha = 0.13;
			this._networkKohonen2.Layers[0].Dim = 0;
            this._networkKohonen2.Layers[0].Learn(1000,input);

			this._networkKohonen2.Compute(DataLoader.ReadInputs("Inputs_Kohonen2.txt"));
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen2.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen2.Outputs[0] + "   " + this._networkKohonen2.Outputs[1]);
			Console.WriteLine(this._networkKohonen2.Outputs[2] + "   " + this._networkKohonen2.Outputs[3]);
        }
		
//		[Test]
		public void TestKohonen_0D_4000E()
        {
			
			/*
			 * 
			 * Alpha = 0.13
			 * sasiedztwo brak
			 * 4000 epok uczenia
			 * 
			 */
			
			List<List<double>> input = new List<List<double>>();
			
			//wektor 1
			List<double> w1 = new List<double> {0, 0, 1, 0, 0, 1, 0, 0, 1};


		    //wektor 2
			List<double> w2 = new List<double> {0, 1, 0, 1, 1, 1, 0, 1, 0};

		    //wektor 3
			List<double> w3 = new List<double> {1, 1, 1, 1, 0, 1, 1, 1, 1};

		    //wekror 4
			List<double> w4 = new List<double> {1, 0, 0, 0, 1, 0, 0, 0, 1};

		    input.Add(w1);
			input.Add(w2);
			input.Add(w3);
			input.Add(w4);
			
			this._networkKohonen2.Layers[0].Alpha = 0.13;
			this._networkKohonen2.Layers[0].Dim = 0;
            this._networkKohonen2.Layers[0].Learn(4000,input);

			this._networkKohonen2.Compute(DataLoader.ReadInputs("Inputs_Kohonen2.txt"));
			Console.WriteLine("wagi:");
			INeuronsCollection neurony = this._networkKohonen2.Layers[0].Neurons;
			
			int counter = 0;
			foreach (INeuron n in neurony)
            {
				foreach (double val in n.Weights)
                {
					Console.Write(val.ToString("##.###") + "  ||   " );
					counter++;
					if(counter % 3 == 0)
						Console.WriteLine("");
				}
				Console.WriteLine("--------");
			}
			
			Console.WriteLine(this._networkKohonen2.Outputs[0] + "   " + this._networkKohonen2.Outputs[1]);
			Console.WriteLine(this._networkKohonen2.Outputs[2] + "   " + this._networkKohonen2.Outputs[3]);
        }
		
		
		
//		[Test]
		public void TestXOR01()
        {
            this._networkXOR01.Compute(DataLoader.ReadInputs("Inputs_XOR_01.txt"));
			
			foreach (double output in this._networkXOR01.Outputs)
            {
				Assert.AreEqual(1,output);
            }
        }
		
//		[Test]
		public void TestXOR10()
        {
            this._networkXOR10.Compute(DataLoader.ReadInputs("Inputs_XOR_10.txt"));
			
			foreach (double output in this._networkXOR10.Outputs)
            {
				Assert.AreEqual(1,output);
            }
        }

 //       [Test]
        public void Test_AND_THRESHOLD_08()
        {
            this._network_Threshold_AND_08.Compute(DataLoader.ReadInputs("Inputs_AND_threshold_08.txt"));

            foreach (double output in this._network_Threshold_AND_08.Outputs)
            {
                Assert.AreEqual(1, output);
            }
        }

		
//		[Test]
		public void TestXOR11()
        {
            this._networkXOR11.Compute(DataLoader.ReadInputs("Inputs_XOR_11.txt"));
			
			foreach (double output in this._networkXOR11.Outputs)
            {
				Assert.AreEqual(0,output);
            }
        }
		
//		[Test]
		public void TestXOR00()
        {
            this._networkXOR00.Compute(DataLoader.ReadInputs("Inputs_XOR_00.txt"));
			
			foreach (double output in this._networkXOR00.Outputs)
            {
				Assert.AreEqual(0,output);
            }
        }
		
//		[Test]
		public void TestAND_00()
        {
            this._network_AND_00.Compute(DataLoader.ReadInputs("Inputs_AND_00.txt"));
			
			foreach (double output in this._network_AND_00.Outputs)
            {
				Assert.AreEqual(output,-0.25);
            }
        }
		
//		[Test]
		public void TestAND_01()
        {
            this._network_AND_01.Compute(DataLoader.ReadInputs("Inputs_AND_01.txt"));
			
			foreach (double output in this._network_AND_01.Outputs)
            {
				Assert.AreEqual(output,0.25);
            }
        }
		
//		[Test]
		public void TestAND_10()
        {
            this._network_AND_10.Compute(DataLoader.ReadInputs("Inputs_AND_10.txt"));
			
			foreach (double output in this._network_AND_10.Outputs)
            {
				Assert.AreEqual(output,0.25);
            }
        }
		
//		[Test]
		public void TestAND_11()
        {
            this._network_AND_11.Compute(DataLoader.ReadInputs("Inputs_AND_11.txt"));
			
			foreach (double output in this._network_AND_11.Outputs)
            {
				Assert.AreEqual(output,0.75);
            }
        }
		
//		[Test]
		public void TestThreshold_11()
        {
            this._network_Threshold_AND_11.Compute(DataLoader.ReadInputs("Inputs_AND_threshold_11.txt"));
			
			foreach (double output in this._network_Threshold_AND_11.Outputs)
            {
				Assert.AreEqual(1.0,output);
            }
        }
		
//		[Test]
		public void TestThreshold_10()
        {
            this._network_Threshold_AND_10.Compute(DataLoader.ReadInputs("Inputs_AND_threshold_10.txt"));
			
			foreach (double output in this._network_Threshold_AND_10.Outputs)
            {
				Assert.AreEqual(0.0,output);
            }
        }
		
//		[Test]
		public void TestThreshold_01()
        {
            this._network_Threshold_AND_01.Compute(DataLoader.ReadInputs("Inputs_AND_threshold_01.txt"));
			
			foreach (double output in this._network_Threshold_AND_01.Outputs)
            {
				Assert.AreEqual(0.0,output);
            }
        }
		
//		[Test]
		public void TestThreshold_00()
        {
            this._network_Threshold_AND_00.Compute(DataLoader.ReadInputs("Inputs_AND_threshold_00.txt"));
			
			foreach (double output in this._network_Threshold_AND_00.Outputs)
            {
				Assert.AreEqual(0.0,output);
            }
        }
		
//        [Test]
        public void TestAfterCompute()
        {
            this._network.Compute(DataLoader.ReadInputs("Inputs.txt"));

            Assert.AreEqual(this._network.Outputs.Count, this._network.Layers[2].Neurons.Count);

            foreach (double output in this._network.Outputs)
            {
                Assert.GreaterOrEqual(output, 0.0);
                Assert.LessOrEqual(output, 1.0);
            }
        }
    }
}