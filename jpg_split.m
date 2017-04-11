function [s1, s2, s3, s4] = jpg_split(BW)
s1 = BW(:, 5:17);
s2 = BW(:, 18:30);
s3 = BW(:, 31:43);
s4 = BW(:, 44:56);
s1 = reshape(s1, 1, 260);
s2 = reshape(s2, 1, 260);
s3 = reshape(s3, 1, 260);
s4 = reshape(s4, 1, 260);
end