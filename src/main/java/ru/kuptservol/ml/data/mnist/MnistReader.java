package ru.kuptservol.ml.data.mnist;

import java.io.ByteArrayOutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

import static java.lang.String.format;

/**
 * @author Sergey Kuptsov
 */
public class MnistReader {
    public static final int LABEL_FILE_MAGIC_NUMBER = 2049;
    public static final int IMAGE_FILE_MAGIC_NUMBER = 2051;

    public static int[][] getLabels(String infile) {

        ByteBuffer bb = loadFileToByteBuffer(infile);

        assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.getInt());

        int numLabels = bb.getInt();
        int[] labels = new int[numLabels];

        for (int i = 0; i < numLabels; ++i) {
            labels[i] = bb.get() & 0xFF; // To unsigned
        }

        int[][] result = new int[numLabels][10];

        for (int i = 0; i < result.length; i++) {
            result[i][labels[i]] = 1;
        }

        return result;
    }

    public static int[][] getImages(String infile) {
        ByteBuffer bb = loadFileToByteBuffer(infile);

        assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());

        int numImages = bb.getInt();
        int numRows = bb.getInt();
        int numColumns = bb.getInt();
        int[][] images = new int[numImages][numRows * numColumns];

        for (int i = 0; i < numImages; i++) {
            readImage(i, images, numRows, numColumns, bb);
        }

        return images;
    }

    private static void readImage(int i, int[][] images, int numRows, int numCols, ByteBuffer bb) {
        for (int j = 0; j < numRows * numCols; j++) {
            int next = bb.get() & 0xFF;
            images[i][j] = next;
        }
    }

    public static void assertMagicNumber(int expectedMagicNumber, int magicNumber) {
        if (expectedMagicNumber != magicNumber) {
            switch (expectedMagicNumber) {
                case LABEL_FILE_MAGIC_NUMBER:
                    throw new RuntimeException("This is not a label file.");
                case IMAGE_FILE_MAGIC_NUMBER:
                    throw new RuntimeException("This is not an image file.");
                default:
                    throw new RuntimeException(
                            format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber));
            }
        }
    }

    public static ByteBuffer loadFileToByteBuffer(String infile) {
        return ByteBuffer.wrap(loadFile(infile));
    }

    public static byte[] loadFile(String infile) {
        try {
            RandomAccessFile f = new RandomAccessFile(infile, "r");
            FileChannel chan = f.getChannel();
            long fileSize = chan.size();
            ByteBuffer bb = ByteBuffer.allocate((int) fileSize);
            chan.read(bb);
            bb.flip();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            for (int i = 0; i < fileSize; i++) {
                baos.write(bb.get());
            }
            chan.close();
            f.close();
            return baos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static String renderImage(int[][] image) {
        StringBuffer sb = new StringBuffer();

        for (int row = 0; row < image.length; row++) {
            sb.append("|");
            for (int col = 0; col < image[row].length; col++) {
                int pixelVal = image[row][col];
                if (pixelVal == 0)
                    sb.append(" ");
                else if (pixelVal < 256 / 3)
                    sb.append(".");
                else if (pixelVal < 2 * (256 / 3))
                    sb.append("x");
                else
                    sb.append("X");
            }
            sb.append("|\n");
        }

        return sb.toString();
    }

    public static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append(s);
        }
        return sb.toString();
    }
}
