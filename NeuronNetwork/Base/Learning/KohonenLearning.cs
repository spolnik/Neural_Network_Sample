using System;
using System.Linq;
using NeuronNetwork.Base.Layers;
using NeuronNetwork.Base.Neurons;

namespace NeuronNetwork.Base.Learning
{
    public class KohonenLearning : IUnsupervisedLearning
    {
        private readonly ILayer _layer;
		
        // network's dimension
		private readonly int _width;

        private double	_learningRate = 0.1;
		private double	_learningRadius = 7;
		
		// squared learning radius multiplied by 2 (precalculated value to speed up computations)
		private double	_squaredRadius2 = 2 * 7 * 7;

		/// <summary>
		/// Learning rate
		/// </summary>
		/// 
		/// <remarks>Determines speed of learning. Value range is [0, 1].
		/// Default value equals to 0.1.</remarks>
		/// 
		public double LearningRate
		{
			get { return this._learningRate; }
			set
			{
				this._learningRate = Math.Max( 0.0, Math.Min( 1.0, value ) );
			}
		}

		/// <summary>
		/// Learning radius
		/// </summary>
		/// 
		/// <remarks>Determines the amount of neurons to be updated around
		/// winner neuron. Neurons, which are in the circle of specified radius,
		/// are updated during the learning procedure. Neurons, which are closer
		/// to the winner neuron, get more update.<br /><br />
		/// Default value equals to 7.</remarks>
		/// 
		public double LearningRadius
		{
			get { return this._learningRadius; }
			set
			{
				this._learningRadius = Math.Max( 0, value );
				this._squaredRadius2 = 2 * this._learningRadius * this._learningRadius;
			}
		}

        public KohonenLearning(ILayer layer)
		{
			int neuronsCount = layer.Neurons.Count;
			this._width = (int) Math.Sqrt( neuronsCount );

            this._layer	= layer;

		}

        public KohonenLearning(ILayer layer, int width)
		{
			this._layer	= layer;
			this._width = width;
		}

		/// <summary>
		/// Runs learning iteration
		/// </summary>
		/// 
		/// <param name="input">input vector</param>
		/// 
		/// <returns>Returns learning error - summary absolute difference between updated
		/// weights and according inputs. The difference is measured according to the neurons
		/// distance to the  winner neuron.</returns>
		/// 
		public double Run( double[] input )
		{
			double error = 0.0;

			this._layer.Compute(input);
			int winner = this._layer.GetWinner( );

			if ( this._learningRadius == 0 )
			{
                INeuron neuron = this._layer.Neurons[winner];

				// update weight of the winner only
			    for (int i = 0; i < neuron.InputsCount; i++)
			        neuron.Weights[i] += (input[i] - neuron.Weights[i]) * this._learningRate;
			}
			else
			{
				// winner's X and Y
				int wx = winner % this._width;
				int wy = winner / this._width;

				for (int j = 0; j < this._layer.Neurons.Count; j++)
				{
                    INeuron neuron = this._layer.Neurons[j];

					int dx = ( j % this._width ) - wx;
					int dy = ( j / this._width ) - wy;

					// update factor ( Gaussian based )
					double factor = Math.Exp( - (double) ( dx * dx + dy * dy ) / this._squaredRadius2 );

					for (int i = 0; i < neuron.InputsCount; i++)
					{
						double e = ( input[i] - neuron.Weights[i] ) * factor;
						error += Math.Abs( e );
						
						neuron.Weights[i] += e * this._learningRate;
					}
				}
			}
			return error;
		}

		/// <summary>
		/// Runs learning epoch
		/// </summary>
		/// 
		/// <param name="input">array of input vectors</param>
		/// 
		/// <returns>Returns summary learning error for the epoch. See <see cref="Run"/>
		/// method for details about learning error calculation.</returns>
		/// 
		public double RunEpoch( double[][] input )
		{
		    return input.Sum(sample => this.Run(sample));
		}
    }
}