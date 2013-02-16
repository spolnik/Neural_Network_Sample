using System.Collections.Generic;

namespace NeuronNetwork.Base
{
    public class OutputsCollection : List<double>
    {
        public OutputsCollection(int count) : base(count)
        {
            this.InitializeElements(count);
        }

        public OutputsCollection(IEnumerable<double> inputs) : base(inputs)
        {
            //Empty constructor
        }

        private void InitializeElements(int count)
        {
            for (int i = 0; i < count; i++)
            {
                this.Add(new double());
            }
        }
    }
}