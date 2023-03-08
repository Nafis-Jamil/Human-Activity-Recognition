package com.example.har_app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.widget.TableRow;
import android.widget.TextView;

import org.json.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private TextView tv1, tv2, tv3, tv4, tv5;
    private TableRow row1, row2, row3, row4, row5;
    private static final int N_SAMPLES = 100;
    private static int index = 0;
    private TextToSpeech tts;
    private SensorManager sensorManager;
    private static Sensor acc, lin_acc, gyro;
    private static List<Float> ax;
    private static List<Float> ay;
    private static List<Float> az;


    private static List<Float> lx;
    private static List<Float> ly;
    private static List<Float> lz;

    private static List<Float> gx;
    private static List<Float> gy;
    private static List<Float> gz;


    private static int counter;

    private static List<Double> result;
    private int count[] = new int[7];
    private static final String[] activities = {"Walking", "stairs", "Jogging", "Sitting", "Standing", "stairs", "Walking"};
    public static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");
    public final String postUrl = "http://192.168.0.110:5000/post";

    //    public final String postUrl = "http://192.168.64.33:5000/post";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        counter = 0;

        tv1 = findViewById(R.id.pred1);
        tv2 = findViewById(R.id.pred2);
        tv3 = findViewById(R.id.pred3);
        tv4 = findViewById(R.id.pred4);
        tv5 = findViewById(R.id.pred5);


        row1 = findViewById(R.id.row1);
        row2 = findViewById(R.id.row2);
        row3 = findViewById(R.id.row3);
        row4 = findViewById(R.id.row4);
        row5 = findViewById(R.id.row5);


        result = new ArrayList<>();
        ax = new ArrayList<>();
        ay = new ArrayList<>();
        az = new ArrayList<>();
        lx = new ArrayList<>();
        ly = new ArrayList<>();
        lz = new ArrayList<>();
        gx = new ArrayList<>();
        gy = new ArrayList<>();
        gz = new ArrayList<>();


        for (int i = 0; i < count.length; i++) {
            count[i] = 0;
        }


        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        lin_acc = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        tts = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int i) {

                // if No error is found then only it will run
                if (i != TextToSpeech.ERROR) {
                    // To Choose language of speech
                    tts.setLanguage(Locale.US);
                }
            }
        });


        setupSensors();

    }


    void setupSensors() {
//        int v = 100;
        sensorManager.registerListener(this, acc, SensorManager.SENSOR_DELAY_FASTEST);
        sensorManager.registerListener(this, lin_acc, SensorManager.SENSOR_DELAY_FASTEST);
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_FASTEST);
    }


    void postRequest(String postUrl, String postBody) throws IOException {

        OkHttpClient client = new OkHttpClient();

        RequestBody body = RequestBody.create(JSON, postBody);

        Request request = new Request.Builder()
                .url(postUrl)
                .post(body)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                call.cancel();
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {

                String rec_data = response.body().string();

                try {
                    JSONObject object = new JSONObject(rec_data);

                    JSONArray array = object.getJSONArray("array");
                    for (int i = 0; i < 7; i++) {
                        result.add(array.getDouble(i));
                    }
                    index = object.getInt("max");


                } catch (JSONException e) {
                    e.printStackTrace();
                }

            }
        });
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        activity_prediction();

        Sensor sensor = sensorEvent.sensor;
        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            ax.add(sensorEvent.values[0]);
            ay.add(sensorEvent.values[1]);
            az.add(sensorEvent.values[2]);

        } else if (sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            lx.add(sensorEvent.values[0]);
            ly.add(sensorEvent.values[1]);
            lz.add(sensorEvent.values[2]);

        } else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gx.add(sensorEvent.values[0]);
            gy.add(sensorEvent.values[1]);
            gz.add(sensorEvent.values[2]);

        }
    }

    private void activity_prediction() {
        if (counter == 10) {
            int maxAt = 0;

            for (int i = 0; i < count.length; i++) {
                maxAt = count[i] > count[maxAt] ? i : maxAt;
            }

            tts.speak(activities[maxAt], TextToSpeech.QUEUE_FLUSH, null, null);

            counter = 0;
            for (int i = 0; i < count.length; i++) {
                count[i] = 0;
            }
        }
        List<Float> data = new ArrayList<>();

        if (ax.size() >= N_SAMPLES && ay.size() >= N_SAMPLES && az.size() >= N_SAMPLES
                && lx.size() >= N_SAMPLES && ly.size() >= N_SAMPLES && lz.size() >= N_SAMPLES
                && gx.size() >= N_SAMPLES && gy.size() >= N_SAMPLES && gz.size() >= N_SAMPLES
        ) {



            data.addAll(ax.subList(0, N_SAMPLES));
            data.addAll(ay.subList(0, N_SAMPLES));
            data.addAll(az.subList(0, N_SAMPLES));

            data.addAll(lx.subList(0, N_SAMPLES));
            data.addAll(ly.subList(0, N_SAMPLES));
            data.addAll(lz.subList(0, N_SAMPLES));

            data.addAll(gx.subList(0, N_SAMPLES));
            data.addAll(gy.subList(0, N_SAMPLES));
            data.addAll(gz.subList(0, N_SAMPLES));



            JSONArray jsonArray = new JSONArray(data);
            final String jsondata = jsonArray.toString();
            try {
                postRequest(postUrl, jsondata);
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (result.size() == 7) {
                count[index]++;
                counter++;
                setValue();
                setColor(index);
            }


            ax.clear();
            ay.clear();
            az.clear();
            lx.clear();
            ly.clear();
            lz.clear();
            gx.clear();
            gy.clear();
            gz.clear();
            result.clear();

        }
    }

    private void setValue() {

        if (result.get(1) > result.get(5)) {
            tv1.setText(Double.toString(result.get(1)));
        } else {
            tv1.setText(Double.toString(result.get(5)));
        }

        tv2.setText(Double.toString(result.get(2)));
        tv3.setText(Double.toString(result.get(3)));
        tv4.setText(Double.toString(result.get(4)));
        if (result.get(0) > result.get(6)) {
            tv5.setText(Double.toString(result.get(0)));
        } else {
            tv5.setText(Double.toString(result.get(6)));
            ;
        }


    }

    private void setColor(int idx) {

        row1.setBackgroundColor(Color.TRANSPARENT);
        row2.setBackgroundColor(Color.TRANSPARENT);
        row3.setBackgroundColor(Color.TRANSPARENT);
        row4.setBackgroundColor(Color.TRANSPARENT);
        row5.setBackgroundColor(Color.TRANSPARENT);


        switch (index) {
            case 0:
                row5.setBackgroundColor(Color.CYAN);
                break;
            case 1:
                row1.setBackgroundColor(Color.CYAN);
                break;
            case 2:
                row2.setBackgroundColor(Color.CYAN);
                break;
            case 3:
                row3.setBackgroundColor(Color.CYAN);
                break;
            case 4:
                row4.setBackgroundColor(Color.CYAN);
                break;
            case 5:
                row1.setBackgroundColor(Color.CYAN);
                break;
            case 6:
                row5.setBackgroundColor(Color.CYAN);
                break;
        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    @Override
    protected void onResume() {
        super.onResume();
        setupSensors();
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        sensorManager.unregisterListener(this);
    }
}