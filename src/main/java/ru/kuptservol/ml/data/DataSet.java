package ru.kuptservol.ml.data;

import java.util.Optional;

import ru.kuptservol.ml.matrix.M;
import lombok.Builder;
import lombok.Getter;

/**
 * @author Sergey Kuptsov
 */
@Builder
@Getter
public class DataSet {
    public M.Data train;
    @Builder.Default
    public Optional<M.Data> validation = Optional.empty();
    @Builder.Default
    public Optional<M.Data> test = Optional.empty();
}
