// generate all tic-tac-toe

// for AI part - want given a board, what move to make
// board:
// input : 9 vector, -1 = O, 0 = blank, 1 = X
// output: 9 vector, 0 = move here is bad, 1 = move here is good
// 
// the initial board

using System.Diagnostics;


// track boards left to process, and those already processes
var unprocessed = new Queue<Board>(); // board states to left to analyze
var processed = new HashSet<ulong>(); // board hashes that have been analyzed
var canonical = new Dictionary<ulong, Board>(); // canonical board for a given hash

// add initial empty board to process list
var root = new Board(size: 3);
canonical.Add(root.Hash(), root);
unprocessed.Enqueue(root);

// while there are more left to process, do them, add new ones to list to be processed
while (unprocessed.Any())
{
    var board = unprocessed.Dequeue();
    var bHash = board.Hash();
    
    // on a previous pass, multiple parents may have added the same child to be processed
    if (processed.Contains(bHash))
        continue; // do nothing more for this one

    // mark board as processed
    processed.Add(bHash);

    if (board.IsWin())
    {
        board.Score = board.XToMove ? -1 : 1;
        continue;
    }

    if (board.IsFull())
    {
        board.Score = 0;
        continue;
    }

    // for each legal move, recurse into children
    var moves = board.GetMoves();
    foreach (var (i,j) in moves)
    {
        // get new board, do move
        var child = new Board(board);
        child.DoMove(i,j);

        // if this child has been seen before, get canonical object
        var cHash = child.Hash();
        if (canonical.ContainsKey(cHash))
            child = canonical[cHash];
        else
            canonical.Add(cHash,child);

        // link children and parents
        board.Children.Add(child);
        child.Parents.Add(board);

        // add to queue - will be discarded elsewhere if already processed
        unprocessed.Enqueue(child);
    }
}

var boards = canonical.Values.ToList();

Console.WriteLine(
    $"Generate: {processed.Count}=5478 boards, "+
    $"{boards.Count(b=>b.IsWin())} wins, "+
    $"{boards.Count(b=>!b.IsWin() && b.IsFull())} draws "+
    $"{boards.Count(b=>b.IsFull())} full"
    );

// how many board with K filled
for (var k = 0; k <= root.size * root.size; ++k)
{
    var tally = boards.Count(b=>b.GetMoves().Count==k);
    Console.WriteLine($"Boards with {k} moves left = {tally}");
}


// recurse on scoring all boards
root.Score = Score(root);
var scoreCount = boards.Count(b=>b.Score != Board.Unscored);
Console.WriteLine($"{scoreCount} boards scored");


int Score(Board b)
{
    foreach (var c in b.Children)
    {
        if (c.Score == Board.Unscored)
            c.Score = Score(c);
    }
    if (b.XToMove)
        return b.Children.Max(c => c.Score);
    else
        return b.Children.Min(c => c.Score);
}

// dump board data to stream
void Output(TextWriter s)
{
    foreach (var b in boards)
    {
        if (b.IsFull() || b.IsWin()) // don't train on these
            continue;

        var size = b.size;
        for (var i = 0; i < size; ++i)
        for (var j = 0; j < size; ++j)
        {
            var v = b.Grid[i,j] switch
            {
                Board.Blank => 0,
                Board.X => 1,
                Board.O => -1,
                _ => throw new NotImplementedException()
            };
            s.Write($"{v},");
        }

        var score = b.Score;

        // ugly, but works :)
        // see if cell k is a move in a child that has the score of this node
        for (var i = 0; i < size; ++i)
        for (var j = 0; j < size; ++j)
        {
            var moveScore = 0; // assume bad move
            foreach (var c in b.Children)
            {
                if (b.Grid[i, j] != c.Grid[i, j] && c.Score == score)
                    moveScore = 1; // nope, this move obtains best score
            }
            s.Write($"{moveScore},");
        }

        s.WriteLine();
    }
}

// dump to screen
//Output(Console.Out);

// save data to file
Save();
void Save()
{
    var fn = "TicTacToeData.txt";
    using (var f = File.CreateText(fn))
        Output(f);
    var lines = File.ReadAllLines(fn).Length;
    Console.WriteLine($"{lines} lines written to {fn}");
}


Console.WriteLine("Done!");
return;

class Board
{
    public override string ToString() => $"{Score}";

    public List<Board> Parents = new();
    public List<Board> Children = new();

    public const int Unscored = Int32.MaxValue;

    // 0 = draw  = full
    // +1 = X just won
    // -1 = O just won
    public int Score = Unscored;

    public ulong Hash()
    {
        ulong hash = 0;
        for (var i = 0; i < size; ++i)
        for (var j = 0; j < size; ++j)
        {
            var t = Grid[i, j] switch
            {
                Blank => 0,
                O=>1, 
                X=>2
            };
            hash = 3 * hash + (ulong)t;
        }

        return hash;
    }
    public Board(Board b)
    {
        this.size = b.size;
        Grid = new char[size, size];
        for (var i = 0; i < size; ++i)
        for (var j = 0; j < size; ++j)
            Grid[i, j] = b.Grid[i,j];
        XToMove = b.XToMove;
        Score = b.Score;
    }
    public Board(int size = 3)
    {
        this.size = size;
        Grid = new char[size,size];
        for (var i = 0; i < size; ++i)
        for (var j = 0; j < size; ++j)
            Grid[i, j] = Blank;
    }

    public void DoMove(int i, int j)
    {
        Trace.Assert(Grid[i,j] == Blank);
        Grid[i, j] = XToMove ? X : O;
        XToMove = !XToMove;
    }

    public bool XToMove = true;

    public const char Blank = '*';
    public const char X = 'X';
    public const char O = 'O';

    public List<(int i,int j)> GetMoves()
    {
        var m = new List<(int, int)>();
        for (var i = 0; i < size; ++i)
        for (var j = 0; j < size; ++j)
        {
            if (Grid[i, j] == Blank)
            {
                m.Add((i,j));
            }
        }

        return m;

    }

    public void Draw()
    {
        for (var i = 0; i < size; ++i)
        {
            for (var j = 0; j < size; ++j)
            {
                Console.Write(Grid[i,j]+" ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }

    public bool IsWin()
    {

        bool winD1 = true, winD2 = true;
        for (var i = 0; i < size; ++i)
        {
            bool winH = true, winV = true;
            for (var j = 0; j < size; ++j)
            {
                winH &= Grid[i, 0] == Grid[i, j];
                winV &= Grid[0, i] == Grid[j, i];
            }

            if (winH && Grid[i, 0] != Blank) return true;
            if (winV && Grid[0, i] != Blank) return true;

            winD1 &= Grid[0, 0] == Grid[i, i];
            winD2 &= Grid[size-1, 0] == Grid[size-1-i, i];

        }
        if (winD1 && Grid[0, 0] != Blank) return true;
        if (winD2 && Grid[size-1, 0] != Blank) return true;

        return false;
    }

    public bool IsFull()
    {
        var full = true;
        for (var i = 0; i < size; ++i)
        for (var j = 0; j < size; ++j)
            full &= Grid[i,j] != Blank;
        return full;
    }


    public int size;
    public char [,] Grid;
}