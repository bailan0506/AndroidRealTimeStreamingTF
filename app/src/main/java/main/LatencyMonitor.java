package main;

/**
 * Created by lkang on 11/6/17.
 */

public class LatencyMonitor {
    private double rtt = 0.0;
    private int rttCounter = 0;

    static final double alpha = 0.3;

    private double onewayLatency = 0.0;
    private int onewayCounter = 0;

    LatencyMonitor() {
    }

    public void recordRTT(long rtt) {
        this.rttCounter++;
        if (this.rttCounter == 1) {
            this.rtt = rtt;
        } else {
            this.rtt = alpha * rtt + (1 - alpha) * this.rtt;
        }
    }

    public void recordOneWayLatency(long oneway) {
        this.onewayCounter++;
        if (this.onewayCounter == 1) {
            this.onewayLatency = oneway;
        } else {
            this.onewayLatency = alpha * oneway + (1 - alpha) * this.onewayLatency;
        }
    }

    public long getAverageRTT() {
        return (long)this.rtt;
    }
    public long getAverageOneWayLatency() {
        return (long)this.onewayLatency;
    }

}
