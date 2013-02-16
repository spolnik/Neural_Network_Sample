using System;

namespace NeuronNetwork.Utility
{
    public static class NeuronNetworkHelper
    {
        public static Random RandomGen
        {
            get
            {
                return new Random((int)DateTime.Now.Ticks);
            }
        }
    }
}