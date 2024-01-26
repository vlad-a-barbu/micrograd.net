<Query Kind="Program">
  <Namespace>static LINQPad.Util</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
  <Namespace>System.Collections.Concurrent</Namespace>
</Query>

#nullable enable

void Main()
{
	var td = ReadData(@"C:\datasets\mnist\mnist_train.csv", 1, 100);    
    var model = new MLP(784, 10);
    
    Train(td, model, 10, 1e-3)
        .Chart(x => x.epoch, x => x.loss.Value, SeriesType.Line)
        .Dump();

	List<(int epoch, Node loss)> Train((Node[], Node)[] td, MLP model, int epochs, double learningRate)
	{
		var losses = new List<(int, Node)>();
		for (var i = 0; i < epochs; i++)
		{
			var loss = CrossEntropyLoss(td, model);
			losses.Add((i, loss));

			loss.Dump($"epoch {i + 1}");

			model.ZeroGrad();
			loss.Backprop();

			model.Update(learningRate);
		}
		return losses;
	}

	Node[] Softmax(Node[] outputs)
	{
		var maxVal = outputs.Max(o => o.Value);
		var exps = outputs.Select(o => (o + -maxVal).Exp()).ToArray();
		var sumExps = exps.Aggregate((acc, v) => acc + v);
		return exps.Select(v => v * (sumExps ^ -1)).ToArray();
	}

	Node CrossEntropyLoss((Node[], Node)[] td, MLP model)
	{
		Node loss = 0;
		foreach (var (inputs, expected) in td)
		{
			var outputs = Softmax(model.Forward(inputs));
			var target = new Node[10].Select((_, index) => index == expected.Value ? 1 : 0).ToArray();
			var epsilon = 1e-10;
			var error = target.Zip(outputs, (t, o) => t * (o + epsilon).Log()).Aggregate((acc, n) => acc + n);
			loss += error * -1;
		}
		return loss * (new Node(1.0 / td.Length));
	}

	(Node[] xs, Node y)[] ReadData(string path, int offset, int nclass)
	{
		var counts = Enumerable.Range(0, 10).ToDictionary(x => x * 1.0, _ => 0);
		var data = new List<(Node[] xs, Node y)>();
		
		using var reader = new StreamReader(File.OpenRead(path));
		var lptr = 0;
		var done = false;
		while (!reader.EndOfStream && !done)
		{
			var line = reader.ReadLine()!;
			if (lptr++ < offset) continue;
			var values = line.Split(',');
			var label = double.Parse(values[0]);
			if (counts[label] < nclass)
			{
				counts[label]++;
				var pixels = values.Skip(1).Select(v => new Node(double.Parse(v))).ToArray();
				data.Add((pixels, label));
			}
			if (counts.Values.All(x => x == nclass)) done = true;
		}
		
		return data.ToArray();
	}
}

/// <summary>
/// Input: int, Outputs: int[] -> [Input, ..Outputs]
/// </summary>
public record MLP(params int[] Sizes)
{
	public Layer[] Layers { get; private set; } =
		Enumerable
			.Range(0, Sizes.Length - 1)
			.Select(i => new Layer(Sizes[i], Sizes[i + 1], i == Sizes.Length - 2))
			.ToArray();
			
	public Node[] Forward(Node[] xs)
	{
		foreach (var layer in Layers)
		{
			xs = layer.Forward(xs);
		}
		
		return xs;
	}

    public IEnumerable<Node> Parameters => Layers.SelectMany(n => n.Parameters);

    public void ZeroGrad()
    {
        foreach (var node in Parameters)
        {
            node.ZeroGrad();
        }
    }
    
    public void Update(double learningRate)
    {
        foreach (var node in Parameters)
        {
            node.Update(learningRate);
        }
    }
}

public record Layer(int In, int Out, bool Linear)
{
	public Neuron[] Neurons { get; private set; } =
		Enumerable
			.Range(0, Out)
			.Select(_ => new Neuron(In, Linear))
			.ToArray();

	public Node[] Forward(Node[] xs)
	{
		var results = new ConcurrentBag<Node>();
		Parallel.ForEach(Neurons, neuron => 
		{
			results.Add(neuron.Forward(xs));
		});
		return results.ToArray();
	}

    public IEnumerable<Node> Parameters => Neurons.SelectMany(n => n.Parameters);
}

public record Neuron(int N, bool Linear)
{
	public Node[] Weights { get; private set; } = 
		Enumerable
			.Range(0, N)
			.Select(_ => new Node(Random.Shared.NextDouble()))
			.ToArray();

	public Node Bias { get; private set; } = Random.Shared.NextDouble();
	
	public Node Forward(Node[] xs)
	{
		var result = Weights.Zip(xs, (w, x) => w * x).Aggregate((acc, n) => acc + n) + Bias;
		
		return Linear ? result : result.ReLU();
	}
    
    public IEnumerable<Node> Parameters => Weights.Append(Bias);
}

public class Node(double Value)
{
    public double Value { get; private set; } = Value;
	public double Gradient { get; private set; } = 0;

	private (Node, Node?)? _parents;
	private OpType? _op;
	private Action? _backprop;

	private Node(double val, (Node, Node?) parents, OpType op) : this(val)
	{
		_parents = parents;
		_op = op;
	}

    public void Update(double learningRate)
    {
        Value -= learningRate * Gradient;
    }

	public void ZeroGrad()
	{
		var visited = new HashSet<Node>();
		var stack = new Stack<Node>();
		stack.Push(this);

		while (stack.Count > 0)
		{
			var node = stack.Pop();
			if (!visited.Contains(node))
			{
				visited.Add(node);
				node.Gradient = 0;

				if (!node._parents.HasValue) continue;
				
				stack.Push(node._parents.Value.Item1);
				if (node._parents.Value.Item2 is not null)
				{
					stack.Push(node._parents.Value.Item2);
				}
			}
		}
	}

	public void Backprop()
	{
		var visited = new HashSet<Node>();
		var sorted = new List<Node>();
		TopologicalSort(this, visited, sorted);

		Gradient = 1;
		foreach (var node in sorted.AsEnumerable().Reverse())
		{
			node._backprop?.Invoke();
		}
	}

	private void TopologicalSort(Node? node, HashSet<Node> visited, List<Node> sorted)
	{
		if (node is null || visited.Contains(node)) return;

		TopologicalSort(node._parents?.Item1, visited, sorted);
		TopologicalSort(node._parents?.Item2, visited, sorted);
		
		visited.Add(node);
		sorted.Add(node);
	}
	
	public Node ReLU()
	{
		var result = new Node(Value < 0 ? 0 : Value, (this, null), OpType.Relu);
		result._backprop = () =>
		{
			Gradient += (result.Value > 0 ? 1 : 0) * result.Gradient;
		};
		return result;
	}

	public Node Log()
	{
		var result = new Node(Math.Log(Value), (this, null), OpType.Log);
		result._backprop = () =>
		{
			Gradient += (1 / Value) * result.Gradient;
		};
		return result;
	}

	public Node Exp()
	{
		var result = new Node(Math.Exp(Value), (this, null), OpType.Exp);
		result._backprop = () => {
			Gradient += result.Value * result.Gradient;
		};
		return result;
	}

	public static Node operator +(Node x, Node y)
	{
		var result = new Node(x.Value + y.Value, (x, y), OpType.Add);
		result._backprop = () => { 
			x.Gradient += result.Gradient;
			y.Gradient += result.Gradient;
		};
		return result;
	}

	public static Node operator *(Node x, Node y)
	{
		var result = new Node(x.Value * y.Value, (x, y), OpType.Mul);
		result._backprop = () =>
		{
			x.Gradient += y.Value * result.Gradient;
			y.Gradient += x.Value * result.Gradient;
		};
		return result;
	}

	public static Node operator ^(Node x, double y)
	{
		var result = new Node(Math.Pow(x.Value, y), (x, null), OpType.Pow);
		result._backprop = () =>
		{
			x.Gradient += (y * (x ^ (y - 1))).Value * result.Gradient;
		};
		return result;
	}

	public static implicit operator Node(double x) => new(x);

	private enum OpType
	{
		Add,
		Mul,
		Pow,
		Exp,
		Relu,
		Log
	}
}
