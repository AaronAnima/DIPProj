import os
import argparse
import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar


def decode_qr(im):
    # Find barcodes and QR codes
    objectt = pyzbar.decode(im)
    success = len(objectt)
    res = objectt[0] if len(objectt) > 0 else 'None'
    return success, res
        
        
def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True
    

def crop_min_rect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite("RAW_Rotate.jpg",img_rot)
    # input()

    # rotate bounding box
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    # fine tune, to clip edges
    delta = 15
    img_crop = img_rot[pts[1][1]+delta:pts[0][1]-delta,
                       pts[1][0]+delta:pts[2][0]-delta]

    return img_crop


def crop_ticket(img):
    thresh, i = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    i = cv2.erode(i, None, iterations=3)
    i = cv2.dilate(i, None, iterations=3)
    contours, heirarchy = cv2.findContours(i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (contours):
        area = [cv2.contourArea(cnt) for cnt in contours]
        index = np.argmax(area)
        cnt = contours[index]
        rect = cv2.minAreaRect(cnt)
        img_croped = crop_min_rect(img, rect)
        return img_croped
    

def crop_qr(img):
    thresh, i = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    i = cv2.bitwise_not(i)
    i = cv2.dilate(i, None, iterations=4)
    i = cv2.erode(i, None, iterations=6)
    contours, heirarchy = cv2.findContours(i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (contours):
        area = [cv2.contourArea(cnt) for cnt in contours]
        index = np.argmax(area)
        cnt = contours[index]
        x, y, w, h = cv2.boundingRect(cnt)
        # fine tune
        delta = 5
        ih, iw = img.shape
        return img[max(0, y - delta):min(y + h + delta, ih),
                   max(0, x - delta):min(x + w + delta, iw)]


def main(img_dir, ticket_dir, qr_dir):
    # crop tickets from train first
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_croped = crop_ticket(img)
        cv2.imwrite(os.path.join(ticket_dir, img_name), img_croped)

    # Then crop QRcode from tickets
    for img_name in os.listdir(ticket_dir):
        img_path = os.path.join(ticket_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        qr_code = crop_qr(img)
        cv2.imwrite(os.path.join(qr_dir, img_name), qr_code)

    # Then crop QRcode from ticket
    acc = 0
    text_file = open("Output.txt", "w")
    for img_name in os.listdir(qr_dir):
        img_path = os.path.join(qr_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        success, res = decode_qr(img)
        acc += success
        text_file.write(img_name + '\t' + str(res))
    print('the decode acc is {:.3f}'.format(acc / 100.))
    print('Results is written in Output.txt')
    text_file.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DIP Final Project')
    # args
    parser.add_argument('--img-dir', default='train', type=str, help='diretory of images')
    parser.add_argument('--ticket-dir', default='tickets', type=str, help='diretory of tickets')
    parser.add_argument('--qr-dir', default='QRs', type=str, help='diretory of QR code')
    
    args = parser.parse_args()
    exists_or_mkdir(args.ticket_dir)
    exists_or_mkdir(args.qr_dir)
    main(args.img_dir, args.ticket_dir, args.qr_dir)
