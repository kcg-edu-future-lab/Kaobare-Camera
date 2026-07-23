namespace KaobareCamera
{
	public class FrameRateManager(double maxFps = 30)
	{
		public double MaxFps { get; } = maxFps;
		public double ActualFps { get; private set; }

		readonly Queue<DateTime> history = new();

		public bool IsAvailable()
		{
			var now = DateTime.Now;

			if (history.Count == 0)
			{
				history.Enqueue(now);
				ActualFps = 0;
				return true;
			}

			var fps = history.Count / (now - history.Peek()).TotalSeconds;
			if (fps > MaxFps) return false;

			history.Enqueue(now);
			ActualFps = fps;

			while (history.Count > 0 && (now - history.Peek()).TotalSeconds > 1)
			{
				history.Dequeue();
			}
			return true;
		}
	}
}
