package utility;

/**
 * Created by wei on 4/18/17.
 */

//make obj for sending HallData
public class ControlCommand {
    public String type;
    public double throttle;
    public double steering;
    public long timeStamp;

    public ControlCommand(double throttle, double steering, long timeStamp){
        //this.type = type;
        this.throttle = throttle;
        this.steering = steering;
        this.timeStamp = timeStamp;
    }
}