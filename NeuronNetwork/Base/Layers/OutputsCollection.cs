using System.Collections;
using System.Collections.Generic;

namespace NeuronNetwork.Base.Layers
{
    public class OutputsCollection : IOutputsCollection
    {
        private readonly List<double> _outputs;

        public OutputsCollection(int count)
        {
            this._outputs = new List<double>(count);
        }

        #region IOutputsCollection Members

        public double this[int index]
        {
            get { return this._outputs[index]; }
            set { this._outputs[index] = value; }
        }

        public bool IsReadOnly
        {
            get { return ((ICollection<double>) this._outputs).IsReadOnly; }
        }

        public int Count
        {
            get { return this._outputs.Count; }
        }

        public bool Remove(double item)
        {
            return this._outputs.Remove(item);
        }

        public void CopyTo(double[] array, int arrayIndex)
        {
            this._outputs.CopyTo(array, arrayIndex);
        }

        public void Clear()
        {
            this._outputs.Clear();
        }

        public bool Contains(double item)
        {
            return this._outputs.Contains(item);
        }

        public void Add(double item)
        {
            this._outputs.Add(item);
        }

        public IEnumerator<double> GetEnumerator()
        {
            return this._outputs.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        #endregion
    }
}