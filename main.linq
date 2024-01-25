<Query Kind="Program" />

#nullable enable

void Main()
{
	Node x1 = 2;
	Node x2 = 0;
	
	Node w1 = -3;
	Node w2 = 1;
	
	Node b = 6.8813735870195432;
	
	var n = x1 * w1 + x2 * w2 + b;
	var e = (2 * n).Exp();
	var o = (e + -1) * ((e + 1) ^ -1);
	
	o.ZeroGrad();
	o.Backprop();
	
	x1.Dump("x1");
	w1.Dump("w1");
	x2.Dump("x2");
	w2.Dump("w2");
}

public record Node(double Value)
{
	public double Gradient { get; private set; } = 0;

	private (Node, Node?)? _parents;
	private OpType? _op;
	private Action? _backprop;

	private Node(double val, (Node, Node?) parents, OpType op) : this(val)
	{
		_parents = parents;
		_op = op;
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
		Exp
	}
}