package ru.kuptservol.jml.data;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.kuptservol.jml.train.listener.LogTrainListener;

/**
 * @author Sergey Kuptsov
 */
public class FilesLoader {
    private final static Logger logger = LoggerFactory.getLogger(LogTrainListener.class);

    public static void downloadIfNotExists(Path pathToDownload, String remoteFileLink, String fileName) throws IOException {
        Path fileDir = Paths.get(pathToDownload.toString(), fileName);
        if (!fileDir.toFile().exists()) {
            if (!pathToDownload.toFile().exists()) {
                logger.info("Creating dir " + pathToDownload);
                java.nio.file.Files.createDirectories(pathToDownload);
            }
            Files.createFile(fileDir);
            logger.info("Downloading from " + remoteFileLink);
            try (
                    ReadableByteChannel readableByteChannel = Channels.newChannel(new URL(remoteFileLink).openStream());
                    FileOutputStream fileOutputStream = new FileOutputStream(fileDir.toFile()))
            {
                fileOutputStream.getChannel().transferFrom(readableByteChannel, 0, Long.MAX_VALUE);
            }
        }
    }

    public static void gunzipIfNotExists(String dir, String gunzipFile, String toFile) throws IOException {
        Path gunzipFilePath = Paths.get(dir, gunzipFile);
        Path toFilePath = Paths.get(dir, toFile);
        if (!toFilePath.toFile().exists()) {
            logger.info("Gunzip from " + gunzipFilePath + " to " + toFilePath);
            try (
                    ReadableByteChannel archiveByteChannel = Channels.newChannel(new GZIPInputStream(new FileInputStream(gunzipFilePath.toFile())));
                    FileOutputStream fileOutputStream = new FileOutputStream(toFilePath.toFile()))
            {
                fileOutputStream.getChannel().transferFrom(archiveByteChannel, 0, Long.MAX_VALUE);
            }
        }
    }
}
